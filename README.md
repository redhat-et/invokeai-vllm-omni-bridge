# invokeai-vllm-omni-bridge

A standalone plugin and client library that integrates [InvokeAI](https://github.com/invoke-ai/InvokeAI) with [vLLM-Omni](https://github.com/vllm-project/vllm) to enable advanced multimodal AI workflows — visual reasoning, image description, style direction, and more — directly inside the InvokeAI node canvas.


## What it does

InvokeAI is a professional-grade generative AI canvas with a composable, node-based workflow engine. vLLM-Omni is a high-throughput multimodal inference server (text + image + audio) exposing an OpenAI-compatible API.

This bridge connects the two: custom InvokeAI nodes act as lightweight clients that send images (and eventually audio) to a running vLLM-Omni server and wire the responses back into the canvas as text prompts or other outputs.

**Example workflow**: sketch → `VisionDescribeNode` → `StyleDirectorNode` → SDXL → photorealistic image.

---

## Repository structure

```
invokeai-vllm-omni-bridge/
├── invokeai_omni_nodes/      # InvokeAI custom node pack (symlinked into ~/invokeai/nodes/)
│   ├── __init__.py
│   ├── config.py             # Environment-based configuration
│   ├── nodes_text.py         # Text chat node
│   └── nodes_vision.py       # Visual reasoning nodes
├── vllm_client/              # Async HTTP client library for vLLM-Omni
│   ├── client.py
│   └── serializers.py
├── charts/                   # Helm chart for OpenShift deployment
│   └── invokeai-omni/
└── pyproject.toml
```

---

## Requirements

- Python 3.11+
- A running [InvokeAI](https://github.com/invoke-ai/InvokeAI) installation (local or container)
- A running vLLM-Omni server with OpenAI-compatible API enabled

---

## Installation (Local Dev Setup)
This project is a custom Node Pack designed to run inside an existing InvokeAI environment.

**Prerequisites:**
### 1. Initialize InvokeAI (if you haven't already):
If you have the InvokeAI python environment installed, run the web server once to generate the necessary directory structure. By default, InvokeAI uses *~/invokeai* as its root.
```bash
invokeai-web --root ~/invokeai
```

### 2. Clone the repository

```bash
git clone https://github.com/redhat-et/invokeai-vllm-omni-bridge.git
cd invokeai-vllm-omni-bridge
```

### 3. Install the package
Make sure your InvokeAI virtual environment is active before running this!

```bash
pip install -e .
# For development (includes pytest, respx):
pip install -e ".[dev]"
```

### 4. Link the node pack into InvokeAI

InvokeAI loads custom plugins from its root nodes folder. To develop locally without copying files back and forth, create a symlink from your Git repository into the InvokeAI root:

```bash
# Note: If your INVOKEAI_ROOT is not ~/invokeai, adjust the destination path accordingly.
ln -s "$(pwd)/invokeai_omni_nodes" ~/invokeai/nodes/invokeai_omni_nodes
```

### 5. Configure environment variables

```bash
export VLLM_BASE_URL="http://localhost:8000/v1"  # vLLM-Omni server URL (include /v1)
export VLLM_API_KEY="EMPTY"                      # API key (EMPTY for unauthenticated servers)
export VLLM_TIMEOUT=120                          # Request timeout in seconds (default: 120)
```

Add these to your shell profile or a `.env` file to persist them.

### 6. Restart InvokeAI

The new nodes will appear in the node palette under the **vLLM-Omni** category.

---

## Available nodes

| Node | Input | Output | Description |
|---|---|---|---|
| `TextChatNode` | Text prompt | Text | Sends a text prompt to a vLLM-Omni model and returns the reply |
| `VisionDescribeNode` | Image + instruction | Text | Describes an image using a vLLM-Omni vision model |

**Coming soon**: `VisualReasoningToPromptNode`, `StyleDirectorNode`, `AudioReasoningNode`

---

## Development

```bash
# Run tests
pytest

# Run a specific test file
pytest tests/test_serializers.py -v
```

---

## Deployment

The `charts/invokeai-omni/` Helm chart deploys the full stack on OpenShift using [KServe](https://kserve.github.io/website/) to serve vLLM-Omni as an `InferenceService`.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
