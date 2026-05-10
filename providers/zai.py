import time

import httpx

from . import PROVIDER_REGISTRY, BaseProvider, ModelInfo


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
            ))
        return models

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
            elif resp.status_code == 429:
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
