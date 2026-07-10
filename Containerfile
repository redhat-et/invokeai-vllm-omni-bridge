FROM ghcr.io/invoke-ai/invokeai:latest

# Install bridge packages into a dedicated directory
COPY vllm_client/ /opt/bridge/vllm_client/
COPY invokeai_omni_nodes/ /opt/bridge/invokeai_omni_nodes/
ENV PYTHONPATH="/opt/bridge:${PYTHONPATH}"

# Register the node pack with InvokeAI's custom-node discovery
COPY invokeai_omni_nodes/ /invokeai/nodes/invokeai_omni_nodes/

ENV VLLM_BASE_URL="" \
    VLLM_IMAGE_BASE_URL="" \
    VLLM_API_KEY="EMPTY" \
    VLLM_TIMEOUT="120"

EXPOSE 9090
