import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def parse_size_b(model_id: str) -> Optional[float]:
    """Parse parameter count (billions) from a model ID."""
    mid = model_id.lower()
    m = re.search(r'(\d+)x(\d+)b', mid)
    if m:
        return float(m.group(1)) * float(m.group(2))
    m = re.search(r'(\d+)b', mid)
    if m:
        return float(m.group(1))
    return None


def parse_model_type(model_id: str) -> str:
    """Classify model from ID into a semantic type label."""
    mid = model_id.lower()
    if 'realtime' in mid:
        return 'realtime'
    if 'embed' in mid:
        return 'embed'
    if 'guard' in mid:
        return 'guardrail'
    if 'r1' in mid or 'reasoning' in mid or 'thinking' in mid:
        return 'reasoning'
    if 'vision' in mid or 'multimodal' in mid:
        return 'vision'
    if 'code' in mid or 'coder' in mid:
        return 'code'
    if 'instruct' in mid or 'chat' in mid:
        return 'instruct'
    return 'general'


@dataclass
class ModelInfo:
    id: str
    api_provider: str
    model_provider: str
    latency_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    total_time_ms: Optional[float] = None
    prefill_ms: Optional[float] = None
    status: str = "pending"
    size_b: Optional[float] = None
    model_type: str = ""


class BaseProvider(ABC):
    name: str = ""
    env_var: str = ""
    base_url: str = ""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get(self.env_var)
        if not key:
            raise ValueError(f"{self.env_var} not set")
        self.api_key = key
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    def _parse_provider(self, model_id: str) -> str:
        return model_id.split("/")[0] if "/" in model_id else "unknown"

    @abstractmethod
    async def get_models(self) -> list[ModelInfo]:
        ...

    async def check_latency(self, model: ModelInfo) -> ModelInfo:
        payload = {
            "model": model.id,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
            "temperature": 0.1,
        }
        try:
            start = time.monotonic()
            resp = await self._client.post("/chat/completions", json=payload)
            wall_ms = (time.monotonic() - start) * 1000

            if resp.status_code == 200:
                self._extract_timing(model, resp, wall_ms)
                model.status = "ok"
            elif resp.status_code in (402, 403):
                model.status = "no_access"
            elif resp.status_code == 429:
                model.status = "rate_limited"
            elif resp.status_code == 400:
                body = resp.text[:100].lower()
                if "not supported" in body or "not found" in body or "not implemented" in body:
                    model.status = "unsupported"
                else:
                    model.status = "error"
            else:
                model.status = f"http_{resp.status_code}"
        except httpx.TimeoutException:
            model.status = "timeout"
        except Exception:
            model.status = "error"
        return model

    def _extract_timing(self, model: ModelInfo, resp: httpx.Response, wall_ms: float) -> None:
        model.total_time_ms = round(wall_ms, 1)
        model.latency_ms = model.total_time_ms
        model.ttft_ms = round(wall_ms, 1)

    async def close(self) -> None:
        await self._client.aclose()


PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {}
