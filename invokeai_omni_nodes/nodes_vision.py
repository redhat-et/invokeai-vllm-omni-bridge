"""Visual reasoning nodes for the invokeai-vllm-omni-bridge node pack.

Nodes in this module send images to a vLLM-Omni multimodal model and return
text descriptions or refined prompts that can be wired into downstream
image-generation nodes.
"""

import asyncio

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
from vllm_client.serializers import image_to_data_url


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
