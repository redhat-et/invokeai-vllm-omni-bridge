"""Visual reasoning nodes for the invokeai-vllm-omni-bridge node pack.

Nodes in this module send images to a vLLM-Omni multimodal model and return
text descriptions or refined prompts that can be wired into downstream
image-generation nodes.
"""

import asyncio
import os

from PIL import Image

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
from vllm_client.serializers import audio_to_data_url, base64_to_pil, image_to_data_url


@invocation_output("vision_describe_output")
class VisionDescribeOutput(BaseInvocationOutput):
    """Output of the VisionDescribeNode — a text description of the input image."""

    description: str = OutputField(description="The model's text description of the image")


@invocation(
    "vision_describe",
    title="vLLM Vision Describe",
    tags=["vllm", "llm", "vision", "image", "describe"],
    category="vLLM-Omni",
    version="1.0.0",
)
class VisionDescribeNode(BaseInvocation):
    """Send an image to a vLLM-Omni multimodal model and return a text description.

    Wire the ``description`` output to any node that accepts a text/prompt
    string — for example, a prompt input for a downstream image-generation node.
    """

    image: ImageField = InputField(description="The image to describe.")
    prompt: str = InputField(
        default="Describe this image in detail.",
        description="Instruction sent alongside the image.",
        ui_component=UIComponent.Textarea,
    )
    model: str = InputField(
        default="",
        description=(
            "Model name as served by vLLM (e.g. 'Qwen/Qwen2-VL-7B-Instruct'). "
            "Leave blank to use the first available model on the server."
        ),
    )

    def invoke(self, context: InvocationContext) -> VisionDescribeOutput:
        """Retrieve the PIL image, encode it, call vLLM-Omni, return the description."""
        pil_image = context.images.get_pil(self.image.image_name)
        data_url = image_to_data_url(pil_image)
        description = asyncio.run(self._describe(data_url))
        return VisionDescribeOutput(description=description)

    async def _describe(self, data_url: str) -> str:
        """Build the multimodal message and call the vLLM client."""
        if not config.base_url:
            raise RuntimeError(
                "VLLM_BASE_URL environment variable is not set. "
                "Export it before starting InvokeAI."
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": self.prompt},
                ],
            }
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

            response = await client.chat_completion(messages=messages, model=model)

        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Unexpected response shape from vLLM: {response}"
            ) from exc


_STYLE_DIRECTOR_SYSTEM_PROMPT = (
    "You are an art director and prompt engineer specialising in AI image generation. "
    "When given an image and a user instruction, extract and amplify the style, mood, and aesthetic "
    "of the image to produce a single, concise style prompt (no more than 80 words). "
    "Focus exclusively on: artistic style, medium (e.g. oil painting, cinematic photography), "
    "lighting (e.g. golden hour, rim light, soft diffused), colour palette, texture, "
    "compositional feel, and overall mood. "
    "Do not describe the subject matter in detail. "
    "Do not include explanations, preamble, or markdown — output only the prompt text."
)

_VISUAL_REASONING_SYSTEM_PROMPT = (
    "You are an expert prompt engineer for text-to-image models such as Stable Diffusion and SDXL. "
    "When given an image and a user instruction, analyse the image carefully and produce a single, "
    "concise image-generation prompt (no more than 100 words). "
    "The prompt must be rich in visual detail: subject, composition, lighting, colour palette, style, "
    "and mood. Do not include explanations, preamble, or markdown — output only the prompt text."
)


@invocation_output("visual_reasoning_to_prompt_output")
class VisualReasoningToPromptOutput(BaseInvocationOutput):
    """Output of the VisualReasoningToPromptNode — a refined image-generation prompt."""

    prompt: str = OutputField(description="A refined text prompt suitable for image generation")


@invocation(
    "visual_reasoning_to_prompt",
    title="vLLM Visual Reasoning to Prompt",
    tags=["vllm", "llm", "vision", "image", "prompt", "reasoning"],
    category="vLLM-Omni",
    version="1.0.0",
)
class VisualReasoningToPromptNode(BaseInvocation):
    """Analyse an image with a user instruction and return a refined image-generation prompt.

    Wire the ``prompt`` output directly into any downstream image-generation node
    (e.g. SDXL, Flux) that accepts a text prompt.

    Example: feed in a rough sketch with the instruction "make this photorealistic"
    and wire the output prompt into a Text-to-Image node.
    """

    image: ImageField = InputField(description="The reference or source image to reason about.")
    instruction: str = InputField(
        default="Turn this into a photorealistic image.",
        description=(
            "What you want to do with the image "
            "(e.g. 'make this photorealistic', 'convert to oil painting style')."
        ),
        ui_component=UIComponent.Textarea,
    )
    model: str = InputField(
        default="",
        description=(
            "Model name as served by vLLM (e.g. 'Qwen/Qwen2-VL-7B-Instruct'). "
            "Leave blank to use the first available model on the server."
        ),
    )

    def invoke(self, context: InvocationContext) -> VisualReasoningToPromptOutput:
        """Retrieve the PIL image, encode it, call vLLM-Omni, return the refined prompt."""
        pil_image = context.images.get_pil(self.image.image_name)
        data_url = image_to_data_url(pil_image)
        prompt = asyncio.run(self._reason(data_url))
        return VisualReasoningToPromptOutput(prompt=prompt)

    async def _reason(self, data_url: str) -> str:
        """Build the multimodal message with a prompt-engineering system prompt and call vLLM."""
        if not config.base_url:
            raise RuntimeError(
                "VLLM_BASE_URL environment variable is not set. "
                "Export it before starting InvokeAI."
            )

        messages = [
            {"role": "system", "content": _VISUAL_REASONING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
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

            response = await client.chat_completion(messages=messages, model=model)

        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Unexpected response shape from vLLM: {response}"
            ) from exc


@invocation_output("style_director_output")
class StyleDirectorOutput(BaseInvocationOutput):
    """Output of the StyleDirectorNode — a style-focused image-generation prompt."""

    prompt: str = OutputField(description="A style-focused prompt suitable for image generation")


@invocation(
    "style_director",
    title="vLLM Style Director",
    tags=["vllm", "llm", "vision", "image", "style", "prompt"],
    category="vLLM-Omni",
    version="1.0.0",
)
class StyleDirectorNode(BaseInvocation):
    """Extract and amplify the style of an image into a rich image-generation prompt.

    Unlike ``VisualReasoningToPromptNode`` (which reasons about content and intent),
    this node focuses exclusively on aesthetic qualities: artistic style, medium,
    lighting, colour palette, texture, and mood.

    Wire the ``prompt`` output into any downstream image-generation node (SDXL, Flux)
    to apply the extracted style to a new generation.
    """

    image: ImageField = InputField(description="The reference image to extract style from.")
    instruction: str = InputField(
        default="Extract the style and mood of this image.",
        description=(
            "Additional direction for the style extraction "
            "(e.g. 'emphasise the lighting', 'focus on colour palette')."
        ),
        ui_component=UIComponent.Textarea,
    )
    model: str = InputField(
        default="",
        description=(
            "Model name as served by vLLM (e.g. 'Qwen/Qwen2-VL-7B-Instruct'). "
            "Leave blank to use the first available model on the server."
        ),
    )

    def invoke(self, context: InvocationContext) -> StyleDirectorOutput:
        """Retrieve the PIL image, encode it, call vLLM-Omni, return the style prompt."""
        pil_image = context.images.get_pil(self.image.image_name)
        data_url = image_to_data_url(pil_image)
        prompt = asyncio.run(self._direct(data_url))
        return StyleDirectorOutput(prompt=prompt)

    async def _direct(self, data_url: str) -> str:
        """Build the multimodal message with the style-direction system prompt and call vLLM."""
        if not config.base_url:
            raise RuntimeError(
                "VLLM_BASE_URL environment variable is not set. "
                "Export it before starting InvokeAI."
            )

        messages = [
            {"role": "system", "content": _STYLE_DIRECTOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
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

            response = await client.chat_completion(messages=messages, model=model)

        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Unexpected response shape from vLLM: {response}"
            ) from exc


# ---------------------------------------------------------------------------
# MultiModalNarratorNode
# ---------------------------------------------------------------------------

_MULTI_MODAL_NARRATOR_SYSTEM_PROMPT = (
    "You are an expert prompt engineer for text-to-image models. "
    "You will receive a sequence of images representing moments in a journey or narrative, "
    "and an audio clip that provides the emotional throughline connecting them. "
    "Do not describe each image separately or blend them together. "
    "Instead, reason about the ARC: what changes across the sequence, what builds, "
    "what does the journey lead toward? Let the audio shape the emotional register of your answer. "
    "Produce a single, concise image-generation prompt (no more than 120 words) that captures "
    "the CULMINATING MOMENT or ESSENCE of this narrative — the image that represents where "
    "this story leads or what it means. This image should show something none of the individual "
    "frames contain directly. "
    "The prompt must be rich in visual detail: subject, composition, lighting, colour palette, "
    "style, and mood. "
    "Do not include explanations, preamble, or markdown — output only the prompt text."
)


@invocation_output("multi_modal_narrator_output")
class MultiModalNarratorOutput(BaseInvocationOutput):
    """Output of MultiModalNarratorNode — a prompt capturing the arc of a visual/audio sequence."""

    prompt: str = OutputField(
        description="Image-generation prompt capturing the culminating moment of the narrative"
    )


@invocation(
    "multi_modal_narrator",
    title="vLLM Multi-Modal Narrator",
    tags=["vllm", "llm", "vision", "audio", "narrative", "sequence", "multimodal"],
    category="vLLM-Omni",
    version="1.0.0",
)
class MultiModalNarratorNode(BaseInvocation):
    """Send a sequence of images and an audio clip to vLLM-Omni in a single request.

    The three images are treated as frames of a journey or narrative — not blended,
    but reasoned about as an arc. The audio provides the emotional throughline.
    The model produces a single prompt capturing the *culminating moment* of the
    sequence: the image that represents where the story leads, showing something
    none of the individual frames contain directly.

    Wire the ``prompt`` output into ``VllmImageGenerationNode`` to generate the
    narrative's conclusion as an image.

    Requires ``VLLM_BASE_URL`` pointing at a vLLM-Omni instance that supports
    image and audio inputs (e.g. ``Qwen/Qwen2.5-Omni-7B``).
    """

    image_1: ImageField = InputField(description="First frame of the sequence (beginning).")
    image_2: ImageField = InputField(description="Second frame of the sequence (middle).")
    image_3: ImageField = InputField(description="Third frame of the sequence (end).")
    audio_path: str = InputField(
        description=(
            "Absolute path to the audio file that provides the emotional throughline "
            "(supported formats: WAV, MP3, OGG, FLAC, M4A)."
        ),
    )
    instruction: str = InputField(
        default="Find the culminating moment or essence of this sequence.",
        description="Additional direction for the narrative compression.",
        ui_component=UIComponent.Textarea,
    )
    model: str = InputField(
        default="",
        description=(
            "Model name as served by vLLM (e.g. 'Qwen/Qwen2.5-Omni-7B'). "
            "Leave blank to use the first available model on the server."
        ),
    )

    def invoke(self, context: InvocationContext) -> MultiModalNarratorOutput:
        """Encode all inputs, send them together to vLLM-Omni, return the narrative prompt."""
        if not os.path.isfile(self.audio_path):
            raise RuntimeError(f"Audio file not found: {self.audio_path}")
        data_urls = [
            image_to_data_url(context.images.get_pil(self.image_1.image_name)),
            image_to_data_url(context.images.get_pil(self.image_2.image_name)),
            image_to_data_url(context.images.get_pil(self.image_3.image_name)),
        ]
        audio_data_url = audio_to_data_url(self.audio_path)
        prompt = asyncio.run(self._narrate(data_urls, audio_data_url))
        return MultiModalNarratorOutput(prompt=prompt)

    async def _narrate(self, image_data_urls: list[str], audio_data_url: str) -> str:
        """Build the interleaved multimodal message and call vLLM-Omni."""
        if not config.base_url:
            raise RuntimeError(
                "VLLM_BASE_URL environment variable is not set. "
                "Export it before starting InvokeAI."
            )

        content: list[dict] = []
        for i, url in enumerate(image_data_urls, start=1):
            content.append({"type": "text", "text": f"Frame {i}:"})
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "audio_url", "audio_url": {"url": audio_data_url}})
        content.append({"type": "text", "text": self.instruction})

        messages = [
            {"role": "system", "content": _MULTI_MODAL_NARRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": content},
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
# VllmImageGenerationNode
# ---------------------------------------------------------------------------

@invocation_output("vllm_image_generation_output")
class VllmImageGenerationOutput(BaseInvocationOutput):
    """Output of VllmImageGenerationNode — a generated image registered in InvokeAI."""

    image: ImageField = OutputField(description="The generated image.")


@invocation(
    "vllm_image_generation",
    title="vLLM Image Generation",
    tags=["vllm", "image", "generation", "flux", "diffusion"],
    category="vLLM-Omni",
    version="1.0.0",
)
class VllmImageGenerationNode(BaseInvocation):
    """Generate an image from a text prompt via vLLM-Omni's image generation endpoint.

    Intended to be served by a Flux model on a separate vLLM-Omni instance.
    Wire the ``prompt`` output from any reasoning node (e.g.
    ``VisualReasoningToPromptNode``) into this node to build a fully
    vLLM-Omni-driven pipeline — no InvokeAI diffusion backend required.

    Requires ``VLLM_IMAGE_BASE_URL`` pointing at the Flux vLLM-Omni instance.
    """

    prompt: str = InputField(
        description="Text prompt describing the image to generate.",
        ui_component=UIComponent.Textarea,
    )
    model: str = InputField(
        default="",
        description=(
            "Model name as served by vLLM (e.g. 'black-forest-labs/FLUX.1-dev'). "
            "Leave blank to use the first available model on the server."
        ),
    )
    width: int = InputField(default=1024, description="Output image width in pixels.")
    height: int = InputField(default=1024, description="Output image height in pixels.")

    def invoke(self, context: InvocationContext) -> VllmImageGenerationOutput:
        """Call vLLM-Omni's image generation endpoint and register the result in InvokeAI."""
        pil_image = asyncio.run(self._generate())
        image_dto = context.images.save(image=pil_image)
        return VllmImageGenerationOutput(image=ImageField(image_name=image_dto.image_name))

    async def _generate(self) -> Image.Image:
        if not config.image_base_url:
            raise RuntimeError(
                "VLLM_IMAGE_BASE_URL environment variable is not set. "
                "Export it before starting InvokeAI."
            )

        size = f"{self.width}x{self.height}"

        async with VllmOmniClient(
            base_url=config.image_base_url,
            api_key=config.api_key,
            timeout=config.timeout,
        ) as client:
            model = self.model.strip()
            if not model:
                models = await client.list_models()
                if not models:
                    raise RuntimeError(
                        "No models found on the vLLM image server and no model name was provided."
                    )
                model = models[0]["id"]

            response = await client.image_generation(
                prompt=self.prompt, model=model, size=size
            )

        try:
            b64_data = response["data"][0]["b64_json"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Unexpected response shape from vLLM: {response}"
            ) from exc

        return base64_to_pil(b64_data)
