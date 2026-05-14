import json
import time

import httpx

from . import PROVIDER_REGISTRY, BaseProvider, ModelInfo, parse_model_type, parse_size_b


class NvidiaProvider(BaseProvider):
    name = "nvidia"
    env_var = "NVIDIA_API_KEY"
    base_url = "https://integrate.api.nvidia.com/v1"

    async def get_models(self) -> list[ModelInfo]:
        resp = await self._client.get("/models")
        resp.raise_for_status()
        data = resp.json()["data"]
        seen: set[str] = set()
        models: list[ModelInfo] = []
        for m in data:
            mid = m["id"]
            if mid not in seen:
                seen.add(mid)
                models.append(ModelInfo(
                    id=mid,
                    api_provider=self.name,
                    model_provider=self._parse_provider(mid),
                    size_b=parse_size_b(mid),
                    model_type=parse_model_type(mid),
                ))
        return models

    async def _do_check(self, model: ModelInfo) -> ModelInfo:
        """Streaming check that also extracts NVIDIA nvext timing from the final SSE chunk."""
        payload = {
            "model": model.id,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
            "temperature": 0.1,
            "stream": True,
        }
        try:
            start = time.monotonic()
            async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.status_code == 200:
                    ttft_recorded = False
                    last_data_line = None
                    async for line in resp.aiter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            if not ttft_recorded:
                                model.ttft_ms = round((time.monotonic() - start) * 1000, 1)
                                ttft_recorded = True
                            last_data_line = line
                    wall_ms = (time.monotonic() - start) * 1000
                    model.total_time_ms = round(wall_ms, 1)
                    model.latency_ms = model.total_time_ms

                    # Try to extract nvext timing from the last SSE chunk
                    if last_data_line:
                        try:
                            chunk = json.loads(last_data_line[6:])  # strip "data: "
                            timing = chunk.get("nvext", {}).get("timing", {})
                            ttft = timing.get("ttft_ms")
                            total = timing.get("total_time_ms")
                            prefill = timing.get("prefill_time_ms")
                            if ttft is not None:
                                model.ttft_ms = round(ttft, 1)
                            if total is not None:
                                model.total_time_ms = round(total, 1)
                                model.latency_ms = model.total_time_ms
                            if prefill is not None:
                                model.prefill_ms = round(prefill, 1)
                        except (json.JSONDecodeError, AttributeError):
                            pass  # fall back to client-side timing

                    if not ttft_recorded:
                        model.ttft_ms = model.total_time_ms
                    model.status = "ok"
                elif resp.status_code in (402, 403):
                    model.status = "no_access"
                elif resp.status_code == 429:
                    model.status = "rate_limited"
                elif resp.status_code == 400:
                    await resp.aread()
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


PROVIDER_REGISTRY["nvidia"] = NvidiaProvider
