"""
Decentralized AI compute node.
Registers on the mesh, advertises available models, serves inference requests.
SDKs: FastAPI, NATS, Ollama, LiteLLM, libsodium, Prometheus
"""
import os
import time
import uuid
import json
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

import httpx
import nats
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, start_http_server

NODE_ID = str(uuid.uuid4())[:12]
INFERENCE_REQUESTS = Counter("inference_requests_total", "Total inference requests", ["model", "status"])
INFERENCE_LATENCY = Histogram("inference_latency_ms", "Inference latency", ["model"])
ACTIVE_MODELS = Gauge("active_models", "Currently loaded models")


@dataclass
class NodeInfo:
    node_id: str
    address: str
    port: int
    models: List[str]
    gpu_available: bool = False
    vram_gb: float = 0.0
    load_factor: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)


class InferenceRequest(BaseModel):
    request_id: str = ""
    model: str = "llama3"
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    encrypted: bool = False
    ciphertext: Optional[str] = None


class ComputeNode:
    """
    A single node in the decentralized inference mesh.
    Advertises models over NATS, serves requests via FastAPI.
    """

    def __init__(
        self,
        port: int = 7000,
        models: Optional[List[str]] = None,
        nats_url: str = "nats://localhost:4222",
        ollama_url: str = "http://localhost:11434",
        metrics_port: int = 9090,
    ):
        self.node_id = NODE_ID
        self.port = port
        self.models = models or ["llama3", "mistral"]
        self.nats_url = nats_url
        self.ollama_url = ollama_url
        self.nc = None
        self.app = FastAPI(title=f"Inference Node {self.node_id}")
        self._setup_routes()
        start_http_server(metrics_port)
        ACTIVE_MODELS.set(len(self.models))
        print(f"[Node {self.node_id}] Ready | models={self.models} | port={port}")

    def _setup_routes(self):
        @self.app.post("/infer")
        async def infer(req: InferenceRequest):
            return await self._handle_inference(req)

        @self.app.get("/health")
        async def health():
            return {"node_id": self.node_id, "models": self.models, "status": "ok"}

        @self.app.get("/models")
        async def list_models():
            return {"node_id": self.node_id, "models": self.models}

    async def _handle_inference(self, req: InferenceRequest) -> Dict:
        if req.model not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {req.model} not available on this node")

        t0 = time.perf_counter()
        INFERENCE_REQUESTS.labels(model=req.model, status="started").inc()

        try:
            # Try Ollama first
            response = await self._call_ollama(req.model, req.prompt, req.max_tokens)
            elapsed = (time.perf_counter() - t0) * 1000
            INFERENCE_LATENCY.labels(model=req.model).observe(elapsed)
            INFERENCE_REQUESTS.labels(model=req.model, status="success").inc()
            return {
                "node_id": self.node_id,
                "request_id": req.request_id or str(uuid.uuid4())[:8],
                "model": req.model,
                "response": response,
                "latency_ms": round(elapsed, 1),
            }
        except Exception as e:
            INFERENCE_REQUESTS.labels(model=req.model, status="error").inc()
            raise HTTPException(status_code=500, detail=str(e))

    async def _call_ollama(self, model: str, prompt: str, max_tokens: int) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.ollama_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False,
                      "options": {"num_predict": max_tokens}},
            )
            resp.raise_for_status()
            return resp.json()["response"]

    async def connect_nats(self):
        try:
            self.nc = await nats.connect(self.nats_url)
            print(f"[Node] NATS connected: {self.nats_url}")
            await self._start_heartbeat()
            await self._subscribe_requests()
        except Exception as e:
            print(f"[Node] NATS unavailable: {e}")

    async def _start_heartbeat(self):
        info = NodeInfo(
            node_id=self.node_id,
            address="localhost",
            port=self.port,
            models=self.models,
        )
        async def beat():
            while True:
                info.last_heartbeat = time.time()
                await self.nc.publish("mesh.nodes.heartbeat", json.dumps(asdict(info)).encode())
                await asyncio.sleep(5)
        asyncio.create_task(beat())

    async def _subscribe_requests(self):
        async def handle(msg):
            req_data = json.loads(msg.data)
            req = InferenceRequest(**req_data)
            result = await self._handle_inference(req)
            if msg.reply:
                await self.nc.publish(msg.reply, json.dumps(result).encode())

        for model in self.models:
            await self.nc.subscribe(f"mesh.infer.{model}", cb=handle)
        print(f"[Node] Subscribed to inference topics for {self.models}")
