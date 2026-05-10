from . import PROVIDER_REGISTRY, BaseProvider, ModelInfo


class GroqProvider(BaseProvider):
    name = "groq"
    env_var = "GROQ_API_KEY"
    base_url = "https://api.groq.com/openai/v1"

    async def get_models(self) -> list[ModelInfo]:
        resp = await self._client.get("/models")
        resp.raise_for_status()
        data = resp.json()["data"]
        models: list[ModelInfo] = []
        for m in data:
            mid = m["id"]
            models.append(ModelInfo(
                id=mid,
                api_provider=self.name,
                model_provider=self._parse_provider(mid),
            ))
        return models


PROVIDER_REGISTRY["groq"] = GroqProvider
