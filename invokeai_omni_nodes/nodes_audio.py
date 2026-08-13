"""Audio reasoning node for the invokeai-vllm-omni-bridge node pack.

The node in this module reads an audio file from disk, sends it to a
vLLM-Omni multimodal model via an ``audio_url`` content part, and returns
a text prompt suitable for wiring into downstream image-generation nodes.

Supported audio formats: WAV, MP3, OGG, FLAC, M4A.
"""

import asyncio
import os

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import InputField, OutputField, UIComponent, ImageField
from invokeai.app.services.shared.invocation_context import InvocationContext

from invokeai_omni_nodes.config import config
from vllm_client.client import VllmOmniClient
from vllm_client.serializers import audio_to_data_url, image_to_data_url


_AUDIO_TO_PROMPT_SYSTEM_PROMPT = (
    "You are an expert prompt engineer for text-to-image models such as Stable Diffusion and SDXL. "
    "When given an audio clip and a user instruction, listen carefully and produce a single, "
    "concise image-generation prompt (no more than 100 words). "
    "The prompt must be rich in visual detail: subject, composition, lighting, colour palette, "
    "style, and mood. "
    "Do not include explanations, preamble, or markdown — output only the prompt text."
)


@invocation_output("audio_to_prompt_output")
class AudioToPromptOutput(BaseInvocationOutput):
    """Output of AudioToPromptNode — a text prompt derived from audio."""

    prompt: str = OutputField(
        description="Image-generation prompt derived from the audio input"
    )


@invocation(
    "audio_to_prompt",
    title="vLLM Audio to Prompt",
    tags=["vllm", "llm", "audio", "prompt"],
    category="vLLM-Omni",
    version="1.0.0",
)
class AudioToPromptNode(BaseInvocation):
    """Send an audio file to vLLM-Omni and return an image-generation prompt.

    The node reads the audio file from the given path, base64-encodes it,
    and sends it to the vLLM-Omni multimodal model as an ``audio_url`` content
    part alongside the instruction.  The model's text response is returned as
    a prompt string that can be wired into any downstream image-generation node.

    Supported formats: WAV, MP3, OGG, FLAC, M4A.
    """

    audio_path: str = InputField(
        description=(
            "Absolute path to the audio file on disk "
            "(supported formats: WAV, MP3, OGG, FLAC, M4A)."
        ),
    )
    instruction: str = InputField(
        default="Describe the scene or mood of this audio as a visual image-generation prompt.",
        description="Instruction sent alongside the audio.",
        ui_component=UIComponent.Textarea,
    )
    model: str = InputField(
        default="",
        description=(
            "Model name as served by vLLM (e.g. 'Qwen/Qwen2.5-Omni-7B'). "
            "Leave blank to use the first available model on the server."
        ),
    )

    def invoke(self, context: InvocationContext) -> AudioToPromptOutput:
        """Read the audio file, encode it, call vLLM-Omni, return the prompt."""
        if not os.path.isfile(self.audio_path):
            raise RuntimeError(f"Audio file not found: {self.audio_path}")
        data_url = audio_to_data_url(self.audio_path)
        prompt = asyncio.run(self._transcribe(data_url))
        return AudioToPromptOutput(prompt=prompt)

    async def _transcribe(self, data_url: str) -> str:
        """Build the multimodal message and call the vLLM client."""
        if not config.base_url:
            raise RuntimeError(
                "VLLM_BASE_URL environment variable is not set. "
                "Export it before starting InvokeAI."
            )

        messages = [
            {"role": "system", "content": _AUDIO_TO_PROMPT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": data_url}},
                    {"type": "text", "text": self.instruction},
                ],
            },
        ]

        async with VllmOmniClient(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
        ) as client:
            model = self.model.strip()
            if not model:
                models = await client.list_models()
                if not models:
                    raise RuntimeError(
                        "No models found on the vLLM server and no model name was provided."
                    )
                model = models[0]["id"]

            response = await client.chat_completion(
                messages=messages, model=model, modalities=["text"]
            )

        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Unexpected response shape from vLLM: {response}"
            ) from exc


# ---------------------------------------------------------------------------
# AudioVisualFusionNode
# ---------------------------------------------------------------------------

_AUDIO_VISUAL_FUSION_SYSTEM_PROMPT = (
    "You are an expert prompt engineer for text-to-image models. "
    "You will receive an image and an audio clip together. "
    "Do not analyse them separately — reason about both simultaneously to understand "
    "the full atmosphere, mood, and meaning that emerges from their combination. "
    "Produce a single, concise image-generation prompt (no more than 100 words) that "
    "captures what neither the image nor the audio could convey alone. "
    "The prompt must be rich in visual detail: subject, composition, lighting, colour palette, "
    "style, and mood. "
    "Do not include explanations, preamble, or markdown — output only the prompt text."
)


@invocation_output("audio_visual_fusion_output")
class AudioVisualFusionOutput(BaseInvocationOutput):
    """Output of AudioVisualFusionNode — a text prompt fusing image and audio."""

    prompt: str = OutputField(
        description="Image-generation prompt derived from the simultaneous fusion of image and audio"
    )


@invocation(
    "audio_visual_fusion",
    title="vLLM Audio-Visual Fusion",
    tags=["vllm", "llm", "audio", "vision", "fusion", "multimodal"],
    category="vLLM-Omni",
    version="1.0.0",
)
class AudioVisualFusionNode(BaseInvocation):
    """Send an image and audio file simultaneously to vLLM-Omni in a single request.

    Unlike chaining separate vision and audio nodes, this node sends both modalities
    together — the model reasons about them at the same time, producing a prompt that
    captures what neither input could convey alone.

    Wire the ``prompt`` output into ``VllmImageGenerationNode`` for a fully
    vLLM-Omni-driven any-to-any pipeline.

    Requires ``VLLM_BASE_URL`` pointing at a vLLM-Omni instance that supports
    both image and audio inputs (e.g. ``Qwen/Qwen2.5-Omni-7B``).
    """

    image: ImageField = InputField(description="The reference image.")
    audio_path: str = InputField(
        description=(
            "Absolute path to the audio file on disk "
            "(supported formats: WAV, MP3, OGG, FLAC, M4A)."
        ),
    )
    instruction: str = InputField(
        default="Fuse the atmosphere of the image and the audio into a single image-generation prompt.",
        description="Additional direction for the fusion.",
        ui_component=UIComponent.Textarea,
    )
    model: str = InputField(
        default="",
        description=(
            "Model name as served by vLLM (e.g. 'Qwen/Qwen2.5-Omni-7B'). "
            "Leave blank to use the first available model on the server."
        ),
    )

    def invoke(self, context: InvocationContext) -> AudioVisualFusionOutput:
        """Encode both inputs, send them together to vLLM-Omni, return the fused prompt."""
        if not os.path.isfile(self.audio_path):
            raise RuntimeError(f"Audio file not found: {self.audio_path}")
        pil_image = context.images.get_pil(self.image.image_name)
        image_data_url = image_to_data_url(pil_image)
        audio_data_url = audio_to_data_url(self.audio_path)
        prompt = asyncio.run(self._fuse(image_data_url, audio_data_url))
        return AudioVisualFusionOutput(prompt=prompt)

    async def _fuse(self, image_data_url: str, audio_data_url: str) -> str:
        """Build the interleaved multimodal message and call vLLM-Omni."""
        if not config.base_url:
            raise RuntimeError(
                "VLLM_BASE_URL environment variable is not set. "
                "Export it before starting InvokeAI."
            )

        messages = [
            {"role": "system", "content": _AUDIO_VISUAL_FUSION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "audio_url", "audio_url": {"url": audio_data_url}},
                    {"type": "text", "text": self.instruction},
                ],
            },
        ]

        async with VllmOmniClient(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
        ) as client:
            model = self.model.strip()
            if not model:
                models = await client.list_models()
                if not models:
                    raise RuntimeError(
                        "No models found on the vLLM server and no model name was provided."
                    )
                model = models[0]["id"]

            response = await client.chat_completion(
                messages=messages, model=model, modalities=["text"]
            )

        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Unexpected response shape from vLLM: {response}"
            ) from exc
