"""Unit tests for vllm_client/serializers.py.

Covers round-trip image encode/decode, data-URL prefix format, JPEG support,
and error handling on invalid input.  No mocking needed — these are pure
function tests.
"""

import base64

import pytest
from PIL import Image

from vllm_client.serializers import base64_to_pil, image_to_base64, image_to_data_url


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_image(mode: str = "RGB", size: tuple[int, int] = (8, 8)) -> Image.Image:
    return Image.new(mode, size, color=(100, 150, 200))


# ---------------------------------------------------------------------------
# image_to_base64
# ---------------------------------------------------------------------------

def test_image_to_base64_returns_string():
    result = image_to_base64(_sample_image())
    assert isinstance(result, str)
    assert len(result) > 0


def test_image_to_base64_is_valid_base64():
    encoded = image_to_base64(_sample_image())
    decoded = base64.b64decode(encoded)  # raises if not valid base64
    assert len(decoded) > 0


def test_image_to_base64_png_magic_bytes():
    encoded = image_to_base64(_sample_image(), format="PNG")
    raw = base64.b64decode(encoded)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_image_to_base64_jpeg_magic_bytes():
    encoded = image_to_base64(_sample_image(), format="JPEG")
    raw = base64.b64decode(encoded)
    assert raw[:2] == b"\xff\xd8"  # JPEG SOI marker


# ---------------------------------------------------------------------------
# image_to_data_url
# ---------------------------------------------------------------------------

def test_image_to_data_url_png_prefix():
    url = image_to_data_url(_sample_image())
    assert url.startswith("data:image/png;base64,")


def test_image_to_data_url_jpeg_prefix():
    url = image_to_data_url(_sample_image(), format="JPEG")
    assert url.startswith("data:image/jpeg;base64,")


def test_image_to_data_url_payload_is_valid_base64():
    url = image_to_data_url(_sample_image())
    _, payload = url.split(",", 1)
    base64.b64decode(payload)  # must not raise


# ---------------------------------------------------------------------------
# base64_to_pil
# ---------------------------------------------------------------------------

def test_round_trip_plain_base64_preserves_size_and_mode():
    original = _sample_image()
    recovered = base64_to_pil(image_to_base64(original))
    assert recovered.size == original.size
    assert recovered.mode == original.mode


def test_round_trip_data_url_preserves_size():
    original = _sample_image()
    recovered = base64_to_pil(image_to_data_url(original))
    assert recovered.size == original.size


def test_round_trip_jpeg_data_url():
    original = _sample_image()
    recovered = base64_to_pil(image_to_data_url(original, format="JPEG"))
    assert recovered.size == original.size


def test_base64_to_pil_strips_data_url_prefix():
    img = _sample_image()
    data_url = image_to_data_url(img, format="PNG")
    assert data_url.startswith("data:")
    recovered = base64_to_pil(data_url)
    assert recovered.size == img.size


def test_base64_to_pil_raises_value_error_on_garbage():
    with pytest.raises(ValueError, match="Failed to decode"):
        base64_to_pil("not-valid-base64!!!")
