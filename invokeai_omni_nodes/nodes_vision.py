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
