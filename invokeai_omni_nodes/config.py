"""Configuration for the invokeai-vllm-omni-bridge node pack.

All settings are read from environment variables so that no secrets
are stored in code.  Set them in your shell or in a `.env` file loaded
by your process manager.

Required environment variables:
    VLLM_BASE_URL   — Base URL of the vLLM-Omni OpenAI-compatible API
                      (e.g. "http://localhost:8000/v1")

Optional environment variables:
    VLLM_API_KEY    — API key sent as a Bearer token (default: "EMPTY",
                      which is the vLLM convention for unauthenticated servers)
    VLLM_TIMEOUT    — Per-request timeout in seconds (default: 120)
"""

import os


def _require(name: str) -> str:
    """Return the value of a required environment variable or raise."""
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set. "
            "Please export it before starting InvokeAI."
        )
    return value


def _optional(name: str, default: str) -> str:
    """Return the value of an optional environment variable or its default."""
    return os.environ.get(name, default)


class BridgeConfig:
    """Singleton-style config object populated from environment variables.

    Instantiate once at import time and import the singleton ``config``
    from this module everywhere else in the package.

    Example::

        from invokeai_omni_nodes.config import config
        client = httpx.AsyncClient(base_url=config.base_url)
    """

    def __init__(self) -> None:
        self.base_url: str = _require("VLLM_BASE_URL")
        self.api_key: str = _optional("VLLM_API_KEY", "EMPTY")
        self.timeout: float = float(_optional("VLLM_TIMEOUT", "120"))

    def __repr__(self) -> str:
        return (
            f"BridgeConfig(base_url={self.base_url!r}, "
            f"api_key={'***' if self.api_key != 'EMPTY' else 'EMPTY'}, "
            f"timeout={self.timeout})"
        )


# Module-level singleton — import this everywhere instead of constructing a new
# BridgeConfig per call.  Evaluation is deferred until first import, so tests
# can set the env vars before importing this module.
config: BridgeConfig = BridgeConfig()