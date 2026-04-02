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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

# conftest.py has already injected the invokeai stubs, so these imports work.
from invokeai.app.invocations.fields import ImageField

from invokeai_omni_nodes.nodes_text import TextChatNode, TextChatOutput
from invokeai_omni_nodes.nodes_vision import (
    StyleDirectorNode,
    StyleDirectorOutput,
    VisionDescribeNode,
    VisionDescribeOutput,
    VisualReasoningToPromptNode,
    VisualReasoningToPromptOutput,
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
