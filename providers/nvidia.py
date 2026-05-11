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

    def _extract_timing(self, model: ModelInfo, resp, wall_ms: float) -> None:
        data = resp.json()
        timing = data.get("nvext", {}).get("timing", {})
        ttft = timing.get("ttft_ms")
        total = timing.get("total_time_ms")
        prefill = timing.get("prefill_time_ms")
        model.ttft_ms = round(ttft, 1) if ttft is not None else round(wall_ms, 1)
        model.total_time_ms = round(total, 1) if total is not None else round(wall_ms, 1)
        model.prefill_ms = round(prefill, 1) if prefill is not None else None
        model.latency_ms = model.total_time_ms


PROVIDER_REGISTRY["nvidia"] = NvidiaProvider
