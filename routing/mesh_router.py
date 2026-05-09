"""
Mesh router: discover nodes via NATS, route requests to best available compute.
SDKs: NATS, httpx, Redis
"""
import json
import time
import random
import asyncio
from typing import Optional, List, Dict, Any

import httpx
import nats
import redis


class MeshRouter:
    """
    Routes inference requests to the best available node in the mesh.
    Discovers nodes via NATS heartbeats, falls back to direct HTTP.
    """

    def __init__(
        self,
        nats_url: str = "nats://localhost:4222",
        redis_url: str = "redis://localhost:6379",
    ):
        self.nats_url = nats_url
        self.nc = None
        self.known_nodes: Dict[str, Dict] = {}
        self._running = False

        try:
            r = redis.from_url(redis_url, decode_responses=True)
            r.ping()
            self._redis = r
            print("[Router] Redis connected for node registry")
        except Exception:
            self._redis = None

    async def connect(self):
        try:
            self.nc = await nats.connect(self.nats_url)
            await self.nc.subscribe("mesh.nodes.heartbeat", cb=self._on_heartbeat)
            self._running = True
            print(f"[Router] Connected to mesh via {self.nats_url}")
        except Exception as e:
            print(f"[Router] NATS unavailable: {e}. Using direct routing.")

    async def _on_heartbeat(self, msg):
        node = json.loads(msg.data)
        nid = node["node_id"]
        self.known_nodes[nid] = node
        if self._redis:
            self._redis.setex(f"mesh:node:{nid}", 30, json.dumps(node))

    def _get_best_node(self, model: str) -> Optional[Dict]:
        """Pick the best available node for a model (lowest load)."""
        candidates = [
            n for n in self.known_nodes.values()
            if model in n.get("models", [])
            and time.time() - n.get("last_heartbeat", 0) < 30
        ]
        if not candidates:
            if self._redis:
                keys = self._redis.keys("mesh:node:*")
                for k in keys:
                    node = json.loads(self._redis.get(k))
                    if model in node.get("models", []):
                        candidates.append(node)
        if not candidates:
            return None
        return min(candidates, key=lambda n: n.get("load_factor", 0))

    async def route(
        self, model: str, prompt: str, max_tokens: int = 512
    ) -> Dict[str, Any]:
        """Route an inference request to the best available node."""
        # Try NATS request-reply first
        if self.nc:
            try:
                payload = json.dumps({
                    "model": model, "prompt": prompt,
                    "max_tokens": max_tokens,
                }).encode()
                resp = await self.nc.request(f"mesh.infer.{model}", payload, timeout=30)
                return json.loads(resp.data)
            except Exception:
                pass

        # Fall back to direct HTTP
        node = self._get_best_node(model)
        if not node:
            # Last resort: try localhost
            node = {"address": "localhost", "port": 7000}

        url = f"http://{node['address']}:{node['port']}/infer"
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json={
                "model": model, "prompt": prompt, "max_tokens": max_tokens
            })
            resp.raise_for_status()
            return resp.json()

    def list_nodes(self) -> List[Dict]:
        return list(self.known_nodes.values())
