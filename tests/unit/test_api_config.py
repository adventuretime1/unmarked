from unmark.api.config import (
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_REWRITE_TIMEOUT_S,
    max_file_bytes,
    rewrite_timeout_s,
)


def test_api_resource_defaults() -> None:
    assert DEFAULT_MAX_FILE_BYTES == 10 * 1024 * 1024
    assert DEFAULT_REWRITE_TIMEOUT_S == 600.0
    assert max_file_bytes({}) == DEFAULT_MAX_FILE_BYTES
    assert rewrite_timeout_s({}) == DEFAULT_REWRITE_TIMEOUT_S


def test_api_resource_limits_accept_environment_overrides() -> None:
    environ = {
        "UNMARK_MAX_FILE_BYTES": "4096",
        "UNMARK_REWRITE_TIMEOUT_S": "12.5",
    }

    assert max_file_bytes(environ) == 4096
    assert rewrite_timeout_s(environ) == 12.5
