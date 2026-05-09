# Decentralized AI Inference Network

Cluster 21 of the NextAura 500 SDKs / 25 Clusters project.

P2P censorship-resistant AI compute mesh. Any node can serve any model. Requests are routed to available compute automatically.

## Architecture

- libp2p for peer-to-peer node discovery and messaging
- IPFS for model artifact storage and retrieval
- Ollama for local LLM serving
- vLLM for high-throughput GPU inference
- LiteLLM for unified multi-provider API
- ONNX Runtime + TensorRT for optimized model serving
- gRPC for high-performance node-to-node RPC
- NATS for event pub/sub across the mesh
- libsodium for end-to-end encrypted inference requests
- WireGuard for zero-trust network overlay

## SDKs Used

libp2p, IPFS, Ollama, vLLM, LiteLLM, TensorRT, ONNX Runtime, gRPC, NATS, libsodium, Vault, WireGuard, ethers.js, The Graph SDK, FastAPI, Redis, Prometheus Client, OpenTelemetry SDK, Docker SDK, Pydantic

## Quickstart

```bash
pip install -r requirements.txt

# Start a compute node
python main.py --mode node --model llama3 --port 7000

# Route a request through the mesh
python main.py --mode request --prompt "Explain quantum entanglement" --model llama3

# Launch mesh dashboard
python main.py --mode dashboard
```
