# ── Stage 1: build wheel ──────────────────────────────────────────────────────
FROM registry.access.redhat.com/ubi9/python-311:latest AS builder

WORKDIR /build

COPY pyproject.toml README.md LICENSE ./
COPY invokeai_omni_nodes/ ./invokeai_omni_nodes/
COPY vllm_client/ ./vllm_client/

RUN pip install --no-cache-dir build \
    && python -m build --wheel --outdir /dist


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM ghcr.io/invoke-ai/invokeai:latest AS runtime

# Install bridge package into InvokeAI's venv
COPY --from=builder /dist/*.whl /tmp/
RUN /invokeai/.venv/bin/pip install --no-cache-dir /tmp/*.whl \
    && rm /tmp/*.whl

# Register the node pack with InvokeAI's custom-node discovery
COPY invokeai_omni_nodes/ /invokeai/nodes/invokeai_omni_nodes/

# Override at deploy time; VLLM_BASE_URL is required
ENV VLLM_BASE_URL="" \
    VLLM_API_KEY="EMPTY" \
    VLLM_TIMEOUT="120"

EXPOSE 9090
