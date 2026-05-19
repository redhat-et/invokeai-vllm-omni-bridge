"""Image serialization helpers for transferring images to and from vLLM-Omni.

These functions operate on :class:`PIL.Image.Image` objects.  When working
with InvokeAI nodes, retrieve the PIL image from the invocation context
*before* calling these helpers::

    pil_image = context.images.get_pil(image_field.image_name)
    data_url = image_to_data_url(pil_image)

The resulting data URL can be embedded directly in an OpenAI-format message::

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
"""

import base64
import io
import os

from PIL import Image


_AUDIO_MIME_TYPES: dict[str, str] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
}


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encode a PIL image to a base64 string.

    Args:
        image: The PIL image to encode.
        format: Image format passed to ``PIL.Image.save`` (e.g. ``"PNG"``,
            ``"JPEG"``).  PNG is the default because it is lossless and
            universally supported.

    Returns:
        A plain base64-encoded string (no ``data:`` prefix).
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def image_to_data_url(image: Image.Image, format: str = "PNG") -> str:
    """Encode a PIL image as a base64 data URL suitable for OpenAI image_url parts.

    Args:
        image: The PIL image to encode.
        format: Image format (``"PNG"`` or ``"JPEG"``).

    Returns:
        A data URL string, e.g. ``"data:image/png;base64,iVBORw0K..."``.
    """
    mime = f"image/{format.lower()}"
    encoded = image_to_base64(image, format=format)
    return f"data:{mime};base64,{encoded}"


def base64_to_pil(data: str) -> Image.Image:
    """Decode a base64 string (with or without a data URL prefix) to a PIL image.

    Args:
        data: A plain base64 string or a data URL (``"data:image/...;base64,..."``).

    Returns:
        A :class:`PIL.Image.Image` object.

    Raises:
        ValueError: If the base64 payload cannot be decoded or is not a valid image.
    """
    if data.startswith("data:"):
        # Strip the "data:<mime>;base64," prefix
        _, encoded = data.split(",", 1)
    else:
        encoded = data

    try:
        raw = base64.b64decode(encoded)
    except Exception as exc:
        raise ValueError(f"Failed to decode base64 payload: {exc}") from exc

    return Image.open(io.BytesIO(raw))


def audio_to_data_url(path: str) -> str:
    """Encode an audio file as a base64 data URL suitable for OpenAI audio_url parts.

    Args:
        path: Absolute path to the audio file on disk.

    Returns:
        A data URL string, e.g. ``"data:audio/wav;base64,UklGRi..."``.

    Raises:
        ValueError: If the file extension is not a supported audio format.
    """
    ext = os.path.splitext(path)[1].lower()
    mime = _AUDIO_MIME_TYPES.get(ext)
    if mime is None:
        supported = ", ".join(_AUDIO_MIME_TYPES)
        raise ValueError(
            f"Unsupported audio format '{ext}'. Supported extensions: {supported}"
        )
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"
