"""vllm_client — async HTTP client for the vLLM-Omni OpenAI-compatible API."""

from vllm_client.client import VllmOmniClient
from vllm_client.serializers import base64_to_pil, image_to_base64, image_to_data_url

__all__ = ["VllmOmniClient", "image_to_base64", "image_to_data_url", "base64_to_pil"]
