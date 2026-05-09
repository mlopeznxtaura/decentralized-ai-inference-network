"""
decentralized-ai-inference-network — Entry Point

P2P AI inference mesh: compute nodes, mesh routing, encrypted requests.

Usage:
  python main.py --mode node --model llama3 --port 7000
  python main.py --mode request --prompt "Hello" --model llama3
  python main.py --mode dashboard
"""
import argparse
import asyncio
import uvicorn


def parse_args():
    parser = argparse.ArgumentParser(description="Decentralized AI Inference Network")
    parser.add_argument("--mode", required=True, choices=["node", "request", "dashboard"])
    parser.add_argument("--model", default="llama3")
    parser.add_argument("--port", type=int, default=7000)
    parser.add_argument("--prompt", default="Explain quantum entanglement in one sentence.")
    parser.add_argument("--nats", default="nats://localhost:4222")
    parser.add_argument("--ollama", default="http://localhost:11434")
    return parser.parse_args()


async def mode_node(args):
    from node.compute_node import ComputeNode
    node = ComputeNode(port=args.port, models=[args.model],
                       nats_url=args.nats, ollama_url=args.ollama)
    await node.connect_nats()
    config = uvicorn.Config(node.app, host="0.0.0.0", port=args.port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


async def mode_request(args):
    from routing.mesh_router import MeshRouter
    router = MeshRouter(nats_url=args.nats)
    await router.connect()
    await asyncio.sleep(2)  # Wait for node discovery
    print(f"[Client] Routing: {args.prompt[:60]}...")
    result = await router.route(args.model, args.prompt)
    print(f"\nResponse from {result.get('node_id', 'node')}:")
    print(result.get("response", result))
    print(f"Latency: {result.get('latency_ms', '?')}ms")


def main():
    args = parse_args()
    print("=" * 55)
    print("  Decentralized AI Inference Network")
    print(f"  Mode: {args.mode.upper()}")
    print("=" * 55)

    if args.mode in ("node", "request"):
        asyncio.run(mode_node(args) if args.mode == "node" else mode_request(args))
    elif args.mode == "dashboard":
        print("Dashboard: http://localhost:8080  (implement with Grafana + Prometheus)")


if __name__ == "__main__":
    main()
