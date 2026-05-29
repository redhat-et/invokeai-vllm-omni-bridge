# invokeai-vllm-omni-bridge

A standalone plugin and client library that integrates [InvokeAI](https://github.com/invoke-ai/InvokeAI) with [vLLM-Omni](https://github.com/vllm-project/vllm) to enable advanced multimodal AI workflows — visual reasoning, image description, style direction, and more — directly inside the InvokeAI node canvas.


## What it does

InvokeAI is a professional-grade generative AI canvas with a composable, node-based workflow engine. vLLM-Omni is a high-throughput multimodal inference server (text + image + audio) exposing an OpenAI-compatible API.

This bridge connects the two: custom InvokeAI nodes act as lightweight clients that send images to a running vLLM-Omni server and wire the responses back into the canvas as text prompts or other outputs.

**Example workflow**: sketch → `VisionDescribeNode` → `StyleDirectorNode` → SDXL → photorealistic image.

---

## Repository structure

```
invokeai-vllm-omni-bridge/
├── invokeai_omni_nodes/      # InvokeAI custom node pack (symlinked into ~/invokeai/nodes/)
│   ├── __init__.py
│   ├── config.py             # Environment-based configuration
│   ├── nodes_text.py         # Text chat node
│   ├── nodes_vision.py       # Visual reasoning nodes
│   └── nodes_audio.py        # Audio-to-prompt node
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
| `VisionDescribeNode` | Image + instruction | Text | Describes an image in natural language |
| `VisualReasoningToPromptNode` | Image + instruction | Text prompt | Reasons about image content and returns a generation prompt |
| `StyleDirectorNode` | Image + instruction | Text prompt | Extracts style/aesthetic from an image and returns a generation prompt |
| `AudioToPromptNode` | Audio file path + instruction | Text prompt | Encodes an audio file and returns an image-generation prompt describing its mood or scene |

All nodes appear in the **vLLM-Omni** category in the InvokeAI node palette.

---

## Development

```bash
# Run tests
pytest

# Run a specific test file
pytest tests/test_serializers.py -v
```

---

## Deployment (OpenShift / KServe)

The `charts/invokeai-omni/` Helm chart deploys the full stack on OpenShift AI using [KServe](https://kserve.github.io/website/) to serve vLLM-Omni as an `InferenceService`.

### Prerequisites

- OpenShift cluster with the **OpenShift AI** operator installed
- KServe enabled (bundled with OpenShift AI)
- A `ServingRuntime` or `ClusterServingRuntime` named `vllm-multimodal` registered in the target namespace
- At least one GPU node with sufficient VRAM (see [GPU requirements](#gpu-requirements) below)
- `helm` CLI ≥ 3.x

### Install

```bash
helm install invokeai-omni charts/invokeai-omni \
  --namespace <your-namespace> \
  --set vllmOmni.modelUri="hf://Qwen/Qwen2.5-Omni-7B" \
  --set invokeai.env.vllmBaseUrl="http://<release-name>-invokeai-omni-vllm-omni-predictor-default:8000/v1"
```

Override `vllmOmni.modelUri` with any HuggingFace model ID supported by your `ServingRuntime`. For initial testing, a smaller quantised variant is recommended to reduce model download time and VRAM requirements.

### Key values

| Value | Default | Description |
|---|---|---|
| `vllmOmni.modelUri` | `hf://Qwen/Qwen2.5-Omni-7B` | HuggingFace model URI for the KServe storage initializer |
| `vllmOmni.runtime` | `vllm-multimodal` | Name of the `ServingRuntime` registered in the cluster |
| `vllmOmni.extraArgs` | `[]` | Extra vLLM engine flags. vLLM-Omni uses a multi-stage engine — use `--stage-overrides` rather than global `--gpu-memory-utilization` / `--max-model-len` flags, which do not propagate to stage engines. Example: `--stage-overrides={"0": {"gpu_memory_utilization": 0.65, "max_model_len": 16384}}` |
| `invokeai.env.vllmBaseUrl` | `http://vllm-omni-predictor-default:8000/v1` | In-cluster URL of the vLLM-Omni predictor |
| `invokeai.image.tag` | `latest` | Bridge container image tag |

### GPU requirements

| Model | Minimum VRAM | Recommended |
|---|---|---|
| Qwen2.5-Omni-7B (fp16) | 40 GB (A100 40 GB with `--stage-overrides`) | 80 GB (H100 / A100 80 GB) |
| Smaller quantised variant (4-bit) | 16 GB | 24 GB |

The chart requests **1 GPU** and **24 Gi memory** for the vLLM-Omni `InferenceService` by default — adjust via `vllmOmni.resources` to match your GPU.

> **Note:** Qwen2.5-Omni-7B uses a three-stage engine (thinker, audio encoder, talker). Total VRAM must accommodate all stages simultaneously; Stage 0 alone requires the bulk of the allocation. Use `--stage-overrides` via `vllmOmni.extraArgs` to tune per-stage memory — the global `--gpu-memory-utilization` flag does not propagate to stage engines.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
