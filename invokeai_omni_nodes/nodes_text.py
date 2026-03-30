"""Text chat node for the invokeai-vllm-omni-bridge node pack.

Sends a text prompt to a vLLM-Omni model and returns the reply as a string,
which can be wired into any downstream InvokeAI node that accepts text
(e.g. a prompt input for image generation).
"""

import asyncio
from typing import Optional

from invokeai.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
    invocation,
    invocation_output,
)
from invokeai.invocations.fields import InputField, OutputField, UIComponent
from pydantic import Field

from invokeai_omni_nodes.config import config
from vllm_client.client import VllmOmniClient


@invocation_output("text_chat_output")
class TextChatOutput(BaseInvocationOutput):
    """Output of the TextChatNode — the model's reply as plain text."""

    reply: str = OutputField(description="The model's text reply")


@invocation(
    "text_chat",
    title="vLLM Text Chat",
    tags=["vllm", "llm", "text", "chat"],
    category="vLLM-Omni",
    version="1.0.0",
)
class TextChatNode(BaseInvocation):
    """Send a text prompt to a vLLM-Omni model and return its reply.

    Wire the ``reply`` output to any node that accepts a text/prompt string —
    for example, a PromptInput or a downstream image-generation node.
    """

    prompt: str = InputField(
        description="The user message to send to the model.",
        ui_component=UIComponent.Textarea,
    )
    model: str = InputField(
        default="",
        description=(
            "Model name as served by vLLM (e.g. 'Qwen/Qwen2.5-7B-Instruct'). "
            "Leave blank to use the first available model on the server."
        ),
    )
    system_prompt: str = InputField(
        default="",
        description="Optional system prompt. Leave blank for no system message.",
        ui_component=UIComponent.Textarea,
    )

    def invoke(self, context: InvocationContext) -> TextChatOutput:
        """Call the vLLM-Omni chat completion endpoint and return the reply."""
        reply = asyncio.run(self._chat())
        return TextChatOutput(reply=reply)

    async def _chat(self) -> str:
        """Build the message list and call the vLLM client."""
        messages: list[dict] = []

        if self.system_prompt.strip():
            messages.append({"role": "system", "content": self.system_prompt.strip()})

        messages.append({"role": "user", "content": self.prompt})

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
