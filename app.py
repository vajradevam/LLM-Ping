import asyncio
import os
import time
import tkinter as tk
from typing import Optional

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Header, Static
from textual.worker import Worker, WorkerState

from providers import PROVIDER_REGISTRY, BaseProvider, ModelInfo
from providers import nvidia, openrouter, groq, zai


def copy_to_clipboard(text: str) -> None:
    """Copy text to system clipboard using tkinter."""
    root = tk.Tk()
    root.withdraw()
    root.clipboard_clear()
    root.clipboard_append(text)
    root.update()
    root.destroy()


def mask_api_key(key: str) -> str:
    """Partially mask an API key, showing first 6 and last 4 chars."""
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:6]}...{key[-4:]}"


class ModelDetailScreen(ModalScreen):
    """Modal screen showing model details with API key."""

    BINDINGS = [("escape", "dismiss", "Close")]

    def __init__(self, model: ModelInfo, api_key: str):
        self.model = model
        self.api_key = api_key
        super().__init__()

    def compose(self) -> ComposeResult:
        with Static(id="model-detail"):
            yield Static(f"[bold]Model:[/bold] {self.model.id}", id="model-id")
            yield Static(f"[bold]Provider:[/bold] {self.model.api_provider}", id="provider")
            yield Static(f"[bold]API Key:[/bold] {mask_api_key(self.api_key)}", id="api-key")
            yield Button("Copy API Key", id="copy-btn", variant="primary")
            yield Button("Close", id="close-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "copy-btn":
            copy_to_clipboard(self.api_key)
            self.app.notify("API Key copied to clipboard")
        elif event.button.id == "close-btn":
            self.dismiss()


class LLMPing(App):
    TITLE = "LLMPing"
    CSS = """
    Screen { background: $surface; }
    DataTable { height: 1fr; margin: 0 1; }
    #status-bar {
        height: 1; padding: 0 1; background: $panel;
        color: $text; content-align: center middle;
    }
    DataTable > .datatable--header {
        background: $primary 20%; color: $text; text-style: bold;
    }
    ModelDetailScreen { align: center middle; }
    #model-detail {
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        width: 60;
        max-width: 80;
    }
    #model-detail Static { margin: 1 0; }
    #model-detail Button { margin: 1 0; width: 100%; }
    """

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("s", "sort_by_latency", "Sort by Latency"),
        Binding("p", "sort_by_api", "Sort by API Provider"),
        Binding("m", "sort_by_model_provider", "Sort by Model Provider"),
        Binding("t", "sort_by_ttft", "Sort by TTFT"),
        Binding("f", "toggle_filter", "Toggle Filter"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable()
        yield Static(id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns(
            "Model", "API Provider", "Model Provider", "Status", "Latency (ms)", "TTFT (ms)"
        )
        self.providers: list[BaseProvider] = []
        self.models: list[ModelInfo] = []
        self.show_all = False
        self.action_refresh()

    def _init_providers(self) -> list[BaseProvider]:
        active: list[BaseProvider] = []
        for name, cls in PROVIDER_REGISTRY.items():
            env_var = cls.env_var
            key = os.environ.get(env_var)
            if not key:
                continue
            try:
                active.append(cls(api_key=key))
            except ValueError:
                continue
        return active

    @work(exclusive=True, group="check", exit_on_error=False)
    async def run_checks(self) -> None:
        table = self.query_one(DataTable)
        status_widget = self.query_one("#status-bar", Static)

        self.providers = self._init_providers()
        if not self.providers:
            status_widget.update(
                "No API providers configured. Set NVIDIA_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY, or ZAI_API_KEY."
            )
            return

        status_widget.update(
            f"Fetching models from {len(self.providers)} provider(s): "
            + ", ".join(p.name for p in self.providers)
        )

        all_models: list[ModelInfo] = []
        for prov in self.providers:
            try:
                ms = await prov.get_models()
                all_models.extend(ms)
            except Exception as e:
                status_widget.update(f"Error fetching models from {prov.name}: {e}")
                return

        self.models = all_models
        total = len(self.models)

        table.clear()
        for m in self.models:
            table.add_row(
                m.id, m.api_provider, m.model_provider, "~", "-", "-", key=self._key(m),
            )

        status_widget.update(f"Checking latency for {total} models across {len(self.providers)} providers ...")

        sem = asyncio.Semaphore(6)
        completed = 0
        ok_count = 0
        start_time = time.monotonic()

        async def check_model(m: ModelInfo) -> None:
            nonlocal completed, ok_count
            async with sem:
                for prov in self.providers:
                    if prov.name == m.api_provider:
                        await prov.check_latency(m)
                        break
                completed += 1
                if m.status == "ok":
                    ok_count += 1
                elapsed = time.monotonic() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                status_widget.update(
                    f"Checked {completed}/{total}  |  "
                    f"{ok_count} available  |  "
                    f"{rate:.1f} models/s"
                )
                self._update_row(m)

        tasks = [check_model(m) for m in self.models]
        await asyncio.gather(*tasks)

        elapsed = time.monotonic() - start_time
        status_widget.update(
            f"Done in {elapsed:.1f}s  |  "
            f"{ok_count}/{total} models accessible  |  "
            "Press [b]r[/] refresh, [b]s[/] sort by latency, [b]f[/] toggle filter"
        )

    @staticmethod
    def _key(m: ModelInfo) -> str:
        return f"{m.api_provider}::{m.id}"

    def _update_row(self, m: ModelInfo) -> None:
        table = self.query_one(DataTable)
        latency_s = f"{m.latency_ms:.0f}" if m.latency_ms is not None else "-"
        ttft_s = f"{m.ttft_ms:.0f}" if m.ttft_ms is not None else "-"
        status_map = {
            "ok": "ok", "pending": "~", "no_access": "no",
            "unsupported": "unsup", "timeout": "timeout", "error": "err", "rate_limited": "rate",
        }
        status_s = status_map.get(m.status, "?")
        try:
            table.update_cell(self._key(m), "Status", status_s)
            table.update_cell(self._key(m), "Latency (ms)", latency_s)
            table.update_cell(self._key(m), "TTFT (ms)", ttft_s)
        except Exception:
            pass

    def _repopulate(self, models: list[ModelInfo]) -> None:
        table = self.query_one(DataTable)
        table.clear()
        for m in models:
            latency_s = f"{m.latency_ms:.0f}" if m.latency_ms is not None else "-"
            ttft_s = f"{m.ttft_ms:.0f}" if m.ttft_ms is not None else "-"
            status_map = {
                "ok": "ok", "pending": "~", "no_access": "no",
                "unsupported": "unsup", "timeout": "timeout", "error": "err", "rate_limited": "rate",
            }
            status_s = status_map.get(m.status, "?")
            table.add_row(
                m.id, m.api_provider, m.model_provider, status_s, latency_s, ttft_s,
                key=self._key(m),
            )

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state == WorkerState.ERROR and event.worker.error:
            self.query_one("#status-bar", Static).update(
                f"Error: {event.worker.error}"
            )

    def action_refresh(self) -> None:
        self.models = []
        self.query_one(DataTable).clear()
        self.run_checks()

    def action_sort_by_latency(self) -> None:
        ok = [m for m in self.models if m.status == "ok" and m.latency_ms is not None]
        ok.sort(key=lambda x: x.latency_ms)
        self._repopulate(ok)
        self.query_one("#status-bar", Static).update(
            f"Showing {len(ok)} accessible models sorted by latency"
        )

    def action_sort_by_api(self) -> None:
        ok = [m for m in self.models if m.status == "ok"]
        ok.sort(key=lambda x: (x.api_provider, x.model_provider, x.id))
        self._repopulate(ok)
        self.query_one("#status-bar", Static).update(
            f"Showing {len(ok)} accessible models sorted by API provider"
        )

    def action_sort_by_model_provider(self) -> None:
        ok = [m for m in self.models if m.status == "ok"]
        ok.sort(key=lambda x: (x.model_provider, x.api_provider, x.id))
        self._repopulate(ok)
        self.query_one("#status-bar", Static).update(
            f"Showing {len(ok)} accessible models sorted by model provider"
        )

    def action_sort_by_ttft(self) -> None:
        ok = [m for m in self.models if m.status == "ok" and m.ttft_ms is not None]
        ok.sort(key=lambda x: x.ttft_ms)
        self._repopulate(ok)
        self.query_one("#status-bar", Static).update(
            f"Showing {len(ok)} accessible models sorted by TTFT"
        )

    def action_toggle_filter(self) -> None:
        self.show_all = not self.show_all
        if self.show_all:
            self._repopulate(self.models)
            self.query_one("#status-bar", Static).update("Showing all models (including inaccessible)")
        else:
            ok = [m for m in self.models if m.status == "ok"]
            self._repopulate(ok)
            self.query_one("#status-bar", Static).update(
                f"Showing {len(ok)} accessible models only"
            )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key:
            key = str(event.row_key.value)
            for m in self.models:
                if self._key(m) == key:
                    self.query_one("#status-bar", Static).update(
                        f"{m.id}  |  API: {m.api_provider}  |  "
                        f"Provider: {m.model_provider}  |  "
                        f"TTFT: {m.ttft_ms or '-'}ms  |  "
                        f"Total: {m.total_time_ms or '-'}ms  |  "
                        f"Status: {m.status}"
                    )
                    break

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        key = str(event.row_key.value)
        model = next((m for m in self.models if self._key(m) == key), None)
        if model:
            provider = next((p for p in self.providers if p.name == model.api_provider), None)
            if provider:
                self.push_screen(ModelDetailScreen(model, provider.api_key))


if __name__ == "__main__":
    LLMPing().run()
