"""Official C2PA verification with a structural, non-verifying fallback."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import cast

from unmark.attachments.common import ai_vendor_in, has_soft_binding
from unmark.attachments.models import (
    AttachmentEvidence,
    AttachmentMediaType,
    EvidenceKind,
    VendorAttribution,
)

# Deliberately empty until genuine Anthropic-signed fixtures establish stable
# issuer + serial/fingerprint identities. A claim-generator string is never added
# here and can never produce ``anthropic_verified`` by itself.
ANTHROPIC_TRUSTED_SIGNERS: frozenset[tuple[str, str | None]] = frozenset()


@dataclass(frozen=True)
class C2paInspection:
    evidence: tuple[AttachmentEvidence, ...]
    verifier: str


def _callable_attribute(value: object, name: str) -> Callable[..., object] | None:
    attribute = getattr(value, name, None)
    return cast("Callable[..., object]", attribute) if callable(attribute) else None


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _active_manifest(store: Mapping[str, object]) -> Mapping[str, object]:
    manifests = _mapping(store.get("manifests"))
    active = store.get("active_manifest")
    return _mapping(manifests.get(str(active))) if active is not None else {}


def _validation_codes(value: object) -> tuple[str, ...]:
    codes: set[str] = set()

    def visit(item: object, depth: int = 0) -> None:
        if depth > 12:
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                normalized = str(key).casefold()
                if normalized in {"code", "status_code"} and isinstance(child, str):
                    codes.add(child)
                else:
                    visit(child, depth + 1)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                visit(child, depth + 1)

    visit(value)
    return tuple(sorted(codes))


def _signature_identity(manifest: Mapping[str, object]) -> tuple[str | None, str | None]:
    signature = _mapping(manifest.get("signature_info"))
    issuer = signature.get("issuer")
    serial = signature.get("cert_serial_number")
    return (
        issuer if isinstance(issuer, str) else None,
        str(serial) if serial is not None else None,
    )


def _anthropic_attribution(
    manifest: Mapping[str, object], *, trusted: bool
) -> tuple[str | None, VendorAttribution]:
    encoded = json.dumps(manifest, ensure_ascii=False, default=str).encode()
    vendor = ai_vendor_in(encoded)
    if vendor != "Anthropic/Claude":
        return vendor, "vendor_unverified" if vendor is not None else "not_applicable"
    issuer, serial = _signature_identity(manifest)
    if trusted and issuer is not None and (issuer.casefold(), serial) in ANTHROPIC_TRUSTED_SIGNERS:
        return vendor, "anthropic_verified"
    return vendor, "vendor_unverified"


def _close(value: object) -> None:
    close = _callable_attribute(value, "close")
    if close is not None:
        close()
        return
    exit_context = _callable_attribute(value, "__exit__")
    if exit_context is not None:
        exit_context(None, None, None)


def inspect_c2pa(
    data: bytes,
    media_type: AttachmentMediaType,
    *,
    embedded_hint: bool,
    source: str | None,
) -> C2paInspection:
    """Validate embedded C2PA with the official reader when it is installed.

    The fallback only reports a structurally located manifest as ``unknown``. It
    never claims signature validity or Anthropic attribution.
    """
    location = source or f"{media_type}:embedded-manifest"
    try:
        module = importlib.import_module("c2pa")
    except ImportError:
        if not embedded_hint:
            return C2paInspection((), "unavailable (install unmark[attachments])")
        return C2paInspection(
            (
                AttachmentEvidence(
                    kind="unknown",
                    source=location,
                    description=(
                        "A C2PA container is present, but the official verifier is not installed; "
                        "signature validity and signer identity are unknown."
                    ),
                    confidence="unknown",
                    removable=True,
                ),
            ),
            "unavailable (install unmark[attachments])",
        )

    reader_type = _callable_attribute(module, "Reader")
    if reader_type is None:
        if not embedded_hint:
            return C2paInspection((), "c2pa-python (Reader unavailable)")
        return C2paInspection(
            (
                AttachmentEvidence(
                    kind="unknown",
                    source=location,
                    description=(
                        "The installed C2PA package does not expose the official Reader API."
                    ),
                    confidence="unknown",
                    removable=True,
                ),
            ),
            "c2pa-python (Reader unavailable)",
        )

    context: object | None = None
    reader: object | None = None
    try:
        context_type = _callable_attribute(module, "Context")
        if context_type is not None:
            from_dict = _callable_attribute(getattr(module, "Context", object()), "from_dict")
            if from_dict is not None:
                context = from_dict(
                    {
                        "verify": {
                            "remote_manifest_fetch": False,
                            "ocsp_fetch": False,
                            "verify_trust": True,
                        }
                    }
                )
        stream = BytesIO(data)
        reader = (
            reader_type(media_type, stream, context=context)
            if context is not None
            else reader_type(media_type, stream)
        )
        render_json = _callable_attribute(reader, "json")
        if render_json is None:
            raise RuntimeError("Reader.json is unavailable")
        raw_store = render_json()
        if not isinstance(raw_store, str):
            raise RuntimeError("Reader.json returned a non-string value")
        decoded: object = json.loads(raw_store)
        store = _mapping(decoded)
        manifest = _active_manifest(store)
        if not manifest:
            if not embedded_hint:
                return C2paInspection((), "c2pa-python official Reader")
            raise RuntimeError("the official reader returned no active manifest")

        state_method = _callable_attribute(reader, "get_validation_state")
        state_value = state_method() if state_method is not None else None
        state = str(state_value).rsplit(".", 1)[-1].casefold() if state_value is not None else ""
        codes = _validation_codes(store.get("validation_status"))
        results_method = _callable_attribute(reader, "get_validation_results")
        if results_method is not None:
            codes = tuple(sorted(set(codes) | set(_validation_codes(results_method()))))
        # The current SDK's ValidationState is authoritative. A valid signature
        # can still carry a non-fatal ``signingCredential.untrusted`` status; that
        # must not be collapsed into a cryptographic failure. Older readers that
        # lack the state API fall back to the legacy validation-status field.
        invalid = state == "invalid" if state else bool(store.get("validation_status"))
        trusted = state == "trusted"
        kind: EvidenceKind = "c2pa_invalid" if invalid else "c2pa_verified"
        vendor, attribution = _anthropic_attribution(manifest, trusted=trusted)
        evidence: list[AttachmentEvidence] = [
            AttachmentEvidence(
                kind=kind,
                source=location,
                description=(
                    "The official C2PA reader reported validation failures."
                    if invalid
                    else (
                        "The official C2PA reader validated the embedded manifest signature "
                        "and bindings."
                    )
                ),
                confidence="cryptographic",
                removable=True,
                vendor=vendor,
                vendor_attribution=attribution,
                validation_state=state or ("invalid" if invalid else "valid"),
                validation_codes=codes,
            )
        ]
        if has_soft_binding(raw_store.encode()):
            evidence.append(
                AttachmentEvidence(
                    kind="soft_binding_declared",
                    source=location,
                    description=(
                        "The C2PA manifest declares a soft or durable binding; no remote "
                        "resolution was attempted."
                    ),
                    confidence="declaration",
                    removable=True,
                    vendor=vendor,
                    vendor_attribution="not_applicable",
                )
            )
        return C2paInspection(tuple(evidence), "c2pa-python official Reader (network disabled)")
    except Exception as exc:
        if not embedded_hint:
            return C2paInspection((), "c2pa-python official Reader (network disabled)")
        return C2paInspection(
            (
                AttachmentEvidence(
                    kind="c2pa_invalid",
                    source=location,
                    description=(
                        "The official C2PA reader could not validate the located manifest: "
                        f"{type(exc).__name__}."
                    ),
                    confidence="cryptographic",
                    removable=True,
                    validation_state="invalid_or_unreadable",
                ),
            ),
            "c2pa-python official Reader (network disabled)",
        )
    finally:
        if reader is not None:
            _close(reader)
        if context is not None:
            _close(context)
