# invokeai-vllm-omni-bridge

A standalone plugin and client library that integrates [InvokeAI](https://github.com/invoke-ai/InvokeAI) with [vLLM-Omni](https://github.com/vllm-project/vllm-omni) to enable advanced multimodal AI workflows — visual reasoning, image description, style direction, audio-to-image, and direct image generation — directly inside the InvokeAI node canvas.


## What it does

InvokeAI is a professional-grade generative AI canvas with a composable, node-based workflow engine. vLLM-Omni is a high-throughput multimodal inference server (text + image + audio) exposing an OpenAI-compatible API.

This bridge connects the two: custom InvokeAI nodes act as lightweight clients for vLLM-Omni, handling multimodal reasoning, prompt generation, and image generation entirely through vLLM-Omni — no InvokeAI diffusion backend required.

**Example workflow (unified pipeline)**: sketch → `VisualReasoningToPromptNode` → `VllmImageGenerationNode` → generated image, all inference through vLLM-Omni.

**Example workflow (audio)**: audio file → `AudioToPromptNode` → `VllmImageGenerationNode` → generated image.

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
export VLLM_BASE_URL="http://localhost:8000/v1"        # vLLM-Omni reasoning/chat instance
export VLLM_IMAGE_BASE_URL="http://localhost:8001/v1"  # vLLM-Omni image generation instance
export VLLM_API_KEY="EMPTY"                            # API key (EMPTY for unauthenticated servers)
export VLLM_TIMEOUT=120                                # Request timeout in seconds (default: 120)
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
| `VllmImageGenerationNode` | Text prompt | Image | Calls vLLM-Omni's image generation endpoint (e.g. Flux), decodes the result, and returns an `ImageField` directly into the InvokeAI canvas |

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
- At least one GPU node with sufficient VRAM (see [GPU requirements](#gpu-requirements) below)
- `helm` CLI ≥ 3.x
- `anyuid` SCC granted to the default service account in your target namespace (required by InvokeAI's entrypoint):
  ```bash
  oc adm policy add-scc-to-user anyuid -z default -n <your-namespace>
  ```

### Install

```bash
helm install invokeai-omni charts/invokeai-omni \
  --namespace <your-namespace> \
  --set vllmOmni.modelUri="hf://Qwen/Qwen2.5-Omni-7B" \
  --set vllmImageGen.modelUri="hf://black-forest-labs/FLUX.2-klein-4B" \
  --set invokeai.env.vllmBaseUrl="http://<release-name>-invokeai-omni-vllm-omni-predictor.<your-namespace>.svc.cluster.local:8000/v1" \
  --set invokeai.env.vllmImageBaseUrl="http://<release-name>-invokeai-omni-vllm-imagegen-predictor.<your-namespace>.svc.cluster.local:8000/v1"
```

Override `vllmOmni.modelUri` and `vllmImageGen.modelUri` with any HuggingFace model IDs supported by your `ServingRuntime`. For initial testing, smaller or quantised variants are recommended to reduce model download time and VRAM requirements.

### Key values

| Value | Default | Description |
|---|---|---|
| `vllmOmni.modelUri` | `hf://Qwen/Qwen2.5-Omni-7B` | HuggingFace model URI for the reasoning ISVC |
| `vllmOmni.runtime` | `vllm-multimodal` | Name of the `ServingRuntime` to use for the reasoning ISVC. The chart creates this runtime automatically. |
| `vllmOmni.extraArgs` | `[]` | Extra vLLM engine flags. vLLM-Omni uses a multi-stage engine — use `--stage-overrides` rather than global `--gpu-memory-utilization` / `--max-model-len` flags, which do not propagate to stage engines. Example: `--stage-overrides={"0": {"gpu_memory_utilization": 0.65, "max_model_len": 16384}}` |
| `vllmImageGen.modelUri` | `hf://black-forest-labs/FLUX.2-klein-4B` | HuggingFace model URI for the image generation ISVC |
| `vllmImageGen.runtime` | `vllm-diffusion` | Name of the `ServingRuntime` to use for the image generation ISVC. The chart creates this runtime automatically. |
| `vllmImageGen.extraArgs` | `[]` | Extra vLLM engine flags for the image generation model |
| `invokeai.env.vllmBaseUrl` | `http://vllm-omni-predictor:8000/v1` | In-cluster URL of the reasoning predictor (KServe RawDeployment mode uses headless Services — use port 8000 directly) |
| `invokeai.env.vllmImageBaseUrl` | `http://vllm-imagegen-predictor:8000/v1` | In-cluster URL of the image generation predictor |
| `invokeai.image.tag` | `latest` | Bridge container image tag |

### GPU requirements

The bridge is model-agnostic — it works with any vLLM-Omni-compatible model served at the configured endpoints. The unified pipeline typically runs two vLLM-Omni instances (one for multimodal reasoning, one for image generation), so GPU resources must cover both simultaneously.

The table below shows validated example configurations:

| Model | Role | Minimum VRAM | Recommended |
|---|---|---|---|
| Qwen2.5-Omni-7B (fp16) | Reasoning (example) | 40 GB (with `--stage-overrides`) | 80 GB (H100 / A100 80 GB) |
| Qwen3-Omni-30B-A3B (MoE, ~3B active) | Reasoning (example) | 2× 80 GB GPUs | 2× H100 80 GB |
| FLUX.2-klein-4B | Image generation (example) | 16 GB | 24 GB |

Any vLLM-Omni-supported multimodal model can be substituted. The chart requests **1 GPU** and **24 Gi memory** for the vLLM-Omni `InferenceService` by default — adjust via `vllmOmni.resources` to match your chosen model and GPU.

> **Note:** Omni-style reasoning models (e.g. Qwen2.5-Omni) use a multi-stage engine (thinker, audio encoder, talker). Total VRAM must accommodate all stages simultaneously; Stage 0 alone requires the bulk of the allocation. Use `--stage-overrides` via `vllmOmni.extraArgs` to tune per-stage memory — the global `--gpu-memory-utilization` flag does not propagate to stage engines.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
