"""Smoke tests for all InvokeAI nodes.

InvokeAI is not installed in the test environment.  ``tests/conftest.py``
stubs out the ``invokeai`` package hierarchy in ``sys.modules`` before pytest
collects this module, so all node imports succeed and node classes behave as
plain pydantic models.

Each node is tested for:
1. Instantiation with valid inputs.
2. A full ``invoke()`` call with the vLLM client mocked — verifies the correct
   output type and content are returned.
3. A ``RuntimeError`` is raised when ``VLLM_BASE_URL`` is not configured.
"""

import struct
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

# conftest.py has already injected the invokeai stubs, so these imports work.
from invokeai.app.invocations.fields import ImageField

from invokeai_omni_nodes.nodes_audio import AudioToPromptNode, AudioToPromptOutput
from invokeai_omni_nodes.nodes_text import TextChatNode, TextChatOutput
from invokeai_omni_nodes.nodes_vision import (
    StyleDirectorNode,
    StyleDirectorOutput,
    VisionDescribeNode,
    VisionDescribeOutput,
    VisualReasoningToPromptNode,
    VisualReasoningToPromptOutput,
    VllmImageGenerationNode,
    VllmImageGenerationOutput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_pil() -> Image.Image:
    return Image.new("RGB", (8, 8), color=(10, 20, 30))


def _make_context(pil_image: Image.Image | None = None) -> MagicMock:
    """Return a minimal mock InvocationContext."""
    ctx = MagicMock()
    if pil_image is not None:
        ctx.images.get_pil.return_value = pil_image
    return ctx


def _make_client_mock(reply: str) -> AsyncMock:
    """Return an AsyncMock that mimics VllmOmniClient as an async context manager.

    ``__aenter__`` is wired to return the mock itself so that
    ``async with VllmOmniClient(...) as client:`` yields this object and
    calls to ``client.chat_completion`` / ``client.list_models`` resolve
    to the configured return values.
    """
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=False)
    mock.list_models.return_value = [{"id": "test-model"}]
    mock.chat_completion.return_value = {
        "choices": [{"message": {"content": reply}}]
    }
    return mock


# ---------------------------------------------------------------------------
# TextChatNode
# ---------------------------------------------------------------------------

class TestTextChatNode:
    def test_instantiation(self):
        node = TextChatNode(prompt="Hello!", model="test-model", system_prompt="")
        assert node.prompt == "Hello!"
        assert node.model == "test-model"

    def test_invoke_returns_text_chat_output(self):
        node = TextChatNode(prompt="Hello!", model="test-model", system_prompt="")
        client_mock = _make_client_mock("Hi from vLLM!")
        with (
            patch("invokeai_omni_nodes.nodes_text.config") as mock_cfg,
            patch("invokeai_omni_nodes.nodes_text.VllmOmniClient", return_value=client_mock),
        ):
            mock_cfg.base_url = "http://localhost:8000/v1"
            mock_cfg.api_key = "EMPTY"
            mock_cfg.timeout = 30.0
            result = node.invoke(_make_context())

        assert isinstance(result, TextChatOutput)
        assert result.reply == "Hi from vLLM!"

    def test_invoke_with_system_prompt(self):
        node = TextChatNode(
            prompt="Hello!", model="test-model", system_prompt="You are helpful."
        )
        client_mock = _make_client_mock("Sure thing!")
        with (
            patch("invokeai_omni_nodes.nodes_text.config") as mock_cfg,
            patch("invokeai_omni_nodes.nodes_text.VllmOmniClient", return_value=client_mock),
        ):
            mock_cfg.base_url = "http://localhost:8000/v1"
            mock_cfg.api_key = "EMPTY"
            mock_cfg.timeout = 30.0
            result = node.invoke(_make_context())

        assert result.reply == "Sure thing!"

    def test_invoke_raises_when_base_url_empty(self):
        node = TextChatNode(prompt="Hello!", model="test-model", system_prompt="")
        with patch("invokeai_omni_nodes.nodes_text.config") as mock_cfg:
            mock_cfg.base_url = ""
            with pytest.raises(RuntimeError, match="VLLM_BASE_URL"):
                node.invoke(_make_context())


# ---------------------------------------------------------------------------
# VisionDescribeNode
# ---------------------------------------------------------------------------

class TestVisionDescribeNode:
    def test_instantiation(self):
        node = VisionDescribeNode(
            image=ImageField(image_name="img.png"),
            prompt="Describe this.",
            model="test-model",
        )
        assert node.image.image_name == "img.png"
        assert node.prompt == "Describe this."

    def test_invoke_returns_vision_describe_output(self):
        node = VisionDescribeNode(
            image=ImageField(image_name="img.png"),
            prompt="Describe this.",
            model="test-model",
        )
        ctx = _make_context(pil_image=_sample_pil())
        client_mock = _make_client_mock("A colourful abstract image.")
        with (
            patch("invokeai_omni_nodes.nodes_vision.config") as mock_cfg,
            patch("invokeai_omni_nodes.nodes_vision.VllmOmniClient", return_value=client_mock),
        ):
            mock_cfg.base_url = "http://localhost:8000/v1"
            mock_cfg.api_key = "EMPTY"
            mock_cfg.timeout = 30.0
            result = node.invoke(ctx)

        assert isinstance(result, VisionDescribeOutput)
        assert result.description == "A colourful abstract image."
        ctx.images.get_pil.assert_called_once_with("img.png")

    def test_invoke_auto_discovers_model_when_blank(self):
        node = VisionDescribeNode(
            image=ImageField(image_name="img.png"),
            prompt="Describe this.",
            model="",  # blank — should trigger list_models()
        )
        ctx = _make_context(pil_image=_sample_pil())
        client_mock = _make_client_mock("An auto-described image.")
        with (
            patch("invokeai_omni_nodes.nodes_vision.config") as mock_cfg,
            patch("invokeai_omni_nodes.nodes_vision.VllmOmniClient", return_value=client_mock),
        ):
            mock_cfg.base_url = "http://localhost:8000/v1"
            mock_cfg.api_key = "EMPTY"
            mock_cfg.timeout = 30.0
            result = node.invoke(ctx)

        assert result.description == "An auto-described image."
        client_mock.list_models.assert_awaited_once()

    def test_invoke_raises_when_base_url_empty(self):
        node = VisionDescribeNode(
            image=ImageField(image_name="img.png"),
            prompt="Describe this.",
            model="test-model",
        )
        ctx = _make_context(pil_image=_sample_pil())
        with patch("invokeai_omni_nodes.nodes_vision.config") as mock_cfg:
            mock_cfg.base_url = ""
            with pytest.raises(RuntimeError, match="VLLM_BASE_URL"):
                node.invoke(ctx)


# ---------------------------------------------------------------------------
# VisualReasoningToPromptNode
# ---------------------------------------------------------------------------

class TestVisualReasoningToPromptNode:
    def test_instantiation(self):
        node = VisualReasoningToPromptNode(
            image=ImageField(image_name="sketch.png"),
            instruction="Make this photorealistic.",
            model="test-model",
        )
        assert node.instruction == "Make this photorealistic."

    def test_invoke_returns_visual_reasoning_output(self):
        node = VisualReasoningToPromptNode(
            image=ImageField(image_name="sketch.png"),
            instruction="Make this photorealistic.",
            model="test-model",
        )
        ctx = _make_context(pil_image=_sample_pil())
        client_mock = _make_client_mock("A photorealistic landscape, golden hour lighting.")
        with (
            patch("invokeai_omni_nodes.nodes_vision.config") as mock_cfg,
            patch("invokeai_omni_nodes.nodes_vision.VllmOmniClient", return_value=client_mock),
        ):
            mock_cfg.base_url = "http://localhost:8000/v1"
            mock_cfg.api_key = "EMPTY"
            mock_cfg.timeout = 30.0
            result = node.invoke(ctx)

        assert isinstance(result, VisualReasoningToPromptOutput)
        assert result.prompt == "A photorealistic landscape, golden hour lighting."

    def test_invoke_raises_when_base_url_empty(self):
        node = VisualReasoningToPromptNode(
            image=ImageField(image_name="sketch.png"),
            instruction="Make this photorealistic.",
            model="test-model",
        )
        ctx = _make_context(pil_image=_sample_pil())
        with patch("invokeai_omni_nodes.nodes_vision.config") as mock_cfg:
            mock_cfg.base_url = ""
            with pytest.raises(RuntimeError, match="VLLM_BASE_URL"):
                node.invoke(ctx)


# ---------------------------------------------------------------------------
# StyleDirectorNode
# ---------------------------------------------------------------------------

class TestStyleDirectorNode:
    def test_instantiation(self):
        node = StyleDirectorNode(
            image=ImageField(image_name="ref.png"),
            instruction="Focus on the lighting.",
            model="test-model",
        )
        assert node.instruction == "Focus on the lighting."

    def test_invoke_returns_style_director_output(self):
        node = StyleDirectorNode(
            image=ImageField(image_name="ref.png"),
            instruction="Focus on the lighting.",
            model="test-model",
        )
        ctx = _make_context(pil_image=_sample_pil())
        client_mock = _make_client_mock("Cinematic lighting, warm tones, film grain.")
        with (
            patch("invokeai_omni_nodes.nodes_vision.config") as mock_cfg,
            patch("invokeai_omni_nodes.nodes_vision.VllmOmniClient", return_value=client_mock),
        ):
            mock_cfg.base_url = "http://localhost:8000/v1"
            mock_cfg.api_key = "EMPTY"
            mock_cfg.timeout = 30.0
            result = node.invoke(ctx)

        assert isinstance(result, StyleDirectorOutput)
        assert result.prompt == "Cinematic lighting, warm tones, film grain."

    def test_invoke_raises_when_base_url_empty(self):
        node = StyleDirectorNode(
            image=ImageField(image_name="ref.png"),
            instruction="Focus on the lighting.",
            model="test-model",
        )
        ctx = _make_context(pil_image=_sample_pil())
        with patch("invokeai_omni_nodes.nodes_vision.config") as mock_cfg:
            mock_cfg.base_url = ""
            with pytest.raises(RuntimeError, match="VLLM_BASE_URL"):
                node.invoke(ctx)


# ---------------------------------------------------------------------------
# VllmImageGenerationNode
# ---------------------------------------------------------------------------

# A 1×1 transparent PNG encoded in base64 — used as a fake vLLM image response.
_FAKE_B64_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
_FAKE_IMAGE_RESPONSE = {"data": [{"b64_json": _FAKE_B64_PNG}]}


def _make_image_client_mock() -> AsyncMock:
    """Return an AsyncMock for VllmOmniClient wired for image generation calls."""
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=False)
    mock.list_models.return_value = [{"id": "flux-model"}]
    mock.image_generation.return_value = _FAKE_IMAGE_RESPONSE
    return mock


class TestVllmImageGenerationNode:
    def test_instantiation(self):
        node = VllmImageGenerationNode(
            prompt="A scenic mountain at dawn.",
            model="black-forest-labs/FLUX.1-dev",
            width=1024,
            height=1024,
        )
        assert node.prompt == "A scenic mountain at dawn."
        assert node.width == 1024

    def test_invoke_returns_image_field(self):
        node = VllmImageGenerationNode(
            prompt="A scenic mountain at dawn.",
            model="flux-model",
            width=512,
            height=512,
        )
        ctx = _make_context()
        ctx.images.save.return_value.image_name = "generated-abc123.png"
        client_mock = _make_image_client_mock()
        with (
            patch("invokeai_omni_nodes.nodes_vision.config") as mock_cfg,
            patch("invokeai_omni_nodes.nodes_vision.VllmOmniClient", return_value=client_mock),
        ):
            mock_cfg.image_base_url = "http://localhost:8001/v1"
            mock_cfg.api_key = "EMPTY"
            mock_cfg.timeout = 30.0
            result = node.invoke(ctx)

        assert isinstance(result, VllmImageGenerationOutput)
        assert result.image.image_name == "generated-abc123.png"
        client_mock.image_generation.assert_awaited_once()

    def test_invoke_passes_correct_size(self):
        node = VllmImageGenerationNode(
            prompt="A forest.", model="flux-model", width=768, height=512
        )
        ctx = _make_context()
        ctx.images.save.return_value.image_name = "out.png"
        client_mock = _make_image_client_mock()
        with (
            patch("invokeai_omni_nodes.nodes_vision.config") as mock_cfg,
            patch("invokeai_omni_nodes.nodes_vision.VllmOmniClient", return_value=client_mock),
        ):
            mock_cfg.image_base_url = "http://localhost:8001/v1"
            mock_cfg.api_key = "EMPTY"
            mock_cfg.timeout = 30.0
            node.invoke(ctx)

        _, kwargs = client_mock.image_generation.call_args
        assert kwargs["size"] == "768x512"

    def test_invoke_auto_discovers_model_when_blank(self):
        node = VllmImageGenerationNode(prompt="A forest.", model="", width=1024, height=1024)
        ctx = _make_context()
        ctx.images.save.return_value.image_name = "out.png"
        client_mock = _make_image_client_mock()
        with (
            patch("invokeai_omni_nodes.nodes_vision.config") as mock_cfg,
            patch("invokeai_omni_nodes.nodes_vision.VllmOmniClient", return_value=client_mock),
        ):
            mock_cfg.image_base_url = "http://localhost:8001/v1"
            mock_cfg.api_key = "EMPTY"
            mock_cfg.timeout = 30.0
            node.invoke(ctx)

        client_mock.list_models.assert_awaited_once()

    def test_invoke_raises_when_image_base_url_empty(self):
        node = VllmImageGenerationNode(
            prompt="A forest.", model="flux-model", width=1024, height=1024
        )
        with patch("invokeai_omni_nodes.nodes_vision.config") as mock_cfg:
            mock_cfg.image_base_url = ""
            with pytest.raises(RuntimeError, match="VLLM_IMAGE_BASE_URL"):
                node.invoke(_make_context())


# ---------------------------------------------------------------------------
# AudioToPromptNode
# ---------------------------------------------------------------------------

def _write_sample_wav(path: str) -> None:
    """Write a minimal valid WAV file (100 silent mono frames at 8 kHz)."""
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(struct.pack("<100h", *([0] * 100)))


class TestAudioToPromptNode:
    def test_instantiation(self):
        node = AudioToPromptNode(
            audio_path="/tmp/sample.wav",
            instruction="Describe this audio.",
            model="test-model",
        )
        assert node.audio_path == "/tmp/sample.wav"
        assert node.instruction == "Describe this audio."

    def test_invoke_returns_audio_to_prompt_output(self, tmp_path):
        wav = str(tmp_path / "sample.wav")
        _write_sample_wav(wav)
        node = AudioToPromptNode(audio_path=wav, instruction="Describe this.", model="test-model")
        client_mock = _make_client_mock("A thunderstorm at dusk, dramatic lighting.")
        with (
            patch("invokeai_omni_nodes.nodes_audio.config") as mock_cfg,
            patch("invokeai_omni_nodes.nodes_audio.VllmOmniClient", return_value=client_mock),
        ):
            mock_cfg.base_url = "http://localhost:8000/v1"
            mock_cfg.api_key = "EMPTY"
            mock_cfg.timeout = 30.0
            result = node.invoke(_make_context())

        assert isinstance(result, AudioToPromptOutput)
        assert result.prompt == "A thunderstorm at dusk, dramatic lighting."

    def test_invoke_auto_discovers_model_when_blank(self, tmp_path):
        wav = str(tmp_path / "sample.wav")
        _write_sample_wav(wav)
        node = AudioToPromptNode(audio_path=wav, instruction="Describe this.", model="")
        client_mock = _make_client_mock("Rain on cobblestones, moody street scene.")
        with (
            patch("invokeai_omni_nodes.nodes_audio.config") as mock_cfg,
            patch("invokeai_omni_nodes.nodes_audio.VllmOmniClient", return_value=client_mock),
        ):
            mock_cfg.base_url = "http://localhost:8000/v1"
            mock_cfg.api_key = "EMPTY"
            mock_cfg.timeout = 30.0
            result = node.invoke(_make_context())

        assert result.prompt == "Rain on cobblestones, moody street scene."
        client_mock.list_models.assert_awaited_once()

    def test_invoke_raises_when_base_url_empty(self, tmp_path):
        wav = str(tmp_path / "sample.wav")
        _write_sample_wav(wav)
        node = AudioToPromptNode(audio_path=wav, instruction="Describe this.", model="test-model")
        with patch("invokeai_omni_nodes.nodes_audio.config") as mock_cfg:
            mock_cfg.base_url = ""
            with pytest.raises(RuntimeError, match="VLLM_BASE_URL"):
                node.invoke(_make_context())

    def test_invoke_raises_when_file_not_found(self):
        node = AudioToPromptNode(
            audio_path="/nonexistent/path/audio.wav",
            instruction="Describe this.",
            model="test-model",
        )
        with pytest.raises(RuntimeError, match="Audio file not found"):
            node.invoke(_make_context())
