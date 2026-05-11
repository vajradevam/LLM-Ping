from . import PROVIDER_REGISTRY, BaseProvider, ModelInfo, parse_model_type, parse_size_b


class OpenRouterProvider(BaseProvider):
    name = "openrouter"
    env_var = "OPENROUTER_API_KEY"
    base_url = "https://openrouter.ai/api/v1"

    async def get_models(self) -> list[ModelInfo]:
        resp = await self._client.get("/models")
        resp.raise_for_status()
        data = resp.json()["data"]
        models: list[ModelInfo] = []
        for m in data:
            mid = m["id"]
            pricing = m.get("pricing", {})
            prompt_price = pricing.get("prompt", "0")
            completion_price = pricing.get("completion", "0")
            if prompt_price == "0" and completion_price == "0":
                arch = m.get("architecture") or {}
                param_count = arch.get("parameter_count")
                size_b = param_count / 1e9 if param_count else parse_size_b(mid)
                models.append(ModelInfo(
                    id=mid,
                    api_provider=self.name,
                    model_provider=self._parse_provider(mid),
                    size_b=size_b,
                    model_type=parse_model_type(mid),
                ))
        return models


PROVIDER_REGISTRY["openrouter"] = OpenRouterProvider
