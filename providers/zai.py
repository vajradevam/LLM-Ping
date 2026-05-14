import time

import httpx

from . import PROVIDER_REGISTRY, BaseProvider, ModelInfo, parse_model_type, parse_size_b


ZAI_KNOWN_MODELS = [
    "glm-5.1", "glm-5-turbo", "glm-5", "glm-4.7", "glm-4.7-flash",
    "glm-4.7-flashx", "glm-4.6", "glm-4.5", "glm-4.5-air",
    "glm-4.5-x", "glm-4.5-airx", "glm-4.5-flash", "glm-4-32b-0414-128k",
    "glm-4.6v-flash",
]


class ZaiProvider(BaseProvider):
    name = "zai"
    env_var = "ZAI_API_KEY"
    base_url = "https://api.z.ai/api/paas/v4"

    async def get_models(self) -> list[ModelInfo]:
        models: list[ModelInfo] = []
        for mid in ZAI_KNOWN_MODELS:
            models.append(ModelInfo(
                id=mid,
                api_provider=self.name,
                model_provider=self._parse_provider(mid),
                size_b=parse_size_b(mid),
                model_type=parse_model_type(mid),
            ))
        return models

    async def _do_check(self, model: ModelInfo) -> ModelInfo:
        """Streaming check with Z.AI-specific 429 error code handling."""
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
                    async for line in resp.aiter_lines():
                        if not ttft_recorded and line.startswith("data: ") and line != "data: [DONE]":
                            model.ttft_ms = round((time.monotonic() - start) * 1000, 1)
                            ttft_recorded = True
                    wall_ms = (time.monotonic() - start) * 1000
                    model.total_time_ms = round(wall_ms, 1)
                    model.latency_ms = model.total_time_ms
                    if not ttft_recorded:
                        model.ttft_ms = model.total_time_ms
                    model.status = "ok"
                elif resp.status_code == 429:
                    # Z.AI uses specific error codes within 429 responses
                    await resp.aread()
                    try:
                        body = resp.json()
                        code = str(body.get("error", {}).get("code", ""))
                    except Exception:
                        code = ""
                    if code == "1305":
                        model.status = "rate_limited"
                    elif code == "1113":
                        model.status = "no_access"
                    else:
                        model.status = "rate_limited"
                elif resp.status_code in (402, 403):
                    model.status = "no_access"
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


PROVIDER_REGISTRY["zai"] = ZaiProvider
