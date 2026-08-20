"""Pseudo-log-likelihood suspicion scoring.

PLL is useful for detector-blind localization, but it is not a watermark detector
and can never establish an official verification state.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from typing import Any, Protocol, cast, runtime_checkable

from unmark.core.errors import DependencyUnavailableError
from unmark.core.spans import StrictModel
from unmark.detectors.localization import TokenAlignment, validate_alignments


class PllTokenScore(StrictModel):
    """Log-likelihood and exact source alignment for one model token."""

    token: TokenAlignment
    log_likelihood: float

    @property
    def suspicion(self) -> float:
        return -self.log_likelihood


@runtime_checkable
class PllBackend(Protocol):
    model_id: str
    model_revision: str
    tokenizer_revision: str
    config_id: str

    def score_tokens(self, text: str) -> tuple[PllTokenScore, ...]: ...


class CachedPllScorer:
    """Thread-safe cache keyed by text and every versioned scoring input."""

    def __init__(self, backend: PllBackend) -> None:
        self.backend = backend
        self._cache: dict[str, tuple[PllTokenScore, ...]] = {}
        self._lock = threading.Lock()

    def cache_key(self, text: str) -> str:
        payload = json.dumps(
            {
                "text": text,
                "model": self.backend.model_id,
                "model_revision": self.backend.model_revision,
                "tokenizer_revision": self.backend.tokenizer_revision,
                "config": self.backend.config_id,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def score_tokens(self, text: str) -> tuple[PllTokenScore, ...]:
        key = self.cache_key(text)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        scores = self.backend.score_tokens(text)
        validate_alignments(text, tuple(item.token for item in scores))
        with self._lock:
            return self._cache.setdefault(key, scores)


class TransformersPllBackend:
    """Optional lazy Hugging Face masked-LM adapter.

    Importing this module never imports Transformers and never downloads a model.
    Construction defaults to ``local_files_only=True``.  Loading occurs on the
    first scoring call so capability discovery remains cheap.
    """

    config_id = "masked-one-at-a-time-v1"

    def __init__(
        self,
        model_id: str,
        *,
        model_revision: str = "main",
        tokenizer_revision: str = "main",
        local_files_only: bool = True,
        loader: Callable[[str, str, str, bool], tuple[object, object]] | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.local_files_only = local_files_only
        self._loader = loader
        self._loaded: tuple[object, object] | None = None

    def _load(self) -> tuple[object, object]:
        if self._loaded is not None:
            return self._loaded
        if self._loader is not None:
            self._loaded = self._loader(
                self.model_id,
                self.model_revision,
                self.tokenizer_revision,
                self.local_files_only,
            )
            return self._loaded
        try:
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForMaskedLM,
                AutoTokenizer,
            )
        except ImportError as error:
            raise DependencyUnavailableError(
                "PLL localization requires the optional 'local-models' dependencies"
            ) from error
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.tokenizer_revision,
            local_files_only=self.local_files_only,
            use_fast=True,
        )
        model = AutoModelForMaskedLM.from_pretrained(
            self.model_id,
            revision=self.model_revision,
            local_files_only=self.local_files_only,
        )
        self._loaded = (tokenizer, model)
        return self._loaded

    def score_tokens(self, text: str) -> tuple[PllTokenScore, ...]:
        """Score with offset mappings; implementation intentionally adapter-local."""
        tokenizer_object, model_object = self._load()
        tokenizer = cast(Any, tokenizer_object)
        model = cast(Any, model_object)
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError as error:
            raise DependencyUnavailableError(
                "PLL localization requires the optional 'local-models' dependencies"
            ) from error

        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        input_ids = encoded["input_ids"][0]
        special_ids = set(tokenizer.all_special_ids)
        mask_id = tokenizer.mask_token_id
        if mask_id is None:
            raise DependencyUnavailableError("selected tokenizer has no mask token")
        results: list[PllTokenScore] = []
        with torch.no_grad():
            for position, token_id in enumerate(input_ids.tolist()):
                start, end = offsets[position]
                if token_id in special_ids or end <= start:
                    continue
                masked = encoded["input_ids"].clone()
                masked[0, position] = mask_id
                logits = model(
                    input_ids=masked, attention_mask=encoded.get("attention_mask")
                ).logits
                log_probability = torch.log_softmax(logits[0, position], dim=-1)[token_id].item()
                results.append(
                    PllTokenScore(
                        token=TokenAlignment(
                            index=len(results),
                            start=start,
                            end=end,
                            text=text[start:end],
                        ),
                        log_likelihood=float(log_probability),
                    )
                )
        return tuple(results)
