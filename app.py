import asyncio
import csv
import os
import shutil
import subprocess
import time
from datetime import datetime
from typing import Optional

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Header, Input, Static
from textual.worker import Worker, WorkerState

from providers import PROVIDER_REGISTRY, BaseProvider, ModelInfo
from providers import nvidia, openrouter, groq, zai


def copy_to_clipboard(text: str) -> None:
    """Copy text to system clipboard using system tools."""
    if shutil.which("xclip"):
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=text.encode(), check=True,
        )
    elif shutil.which("xsel"):
        subprocess.run(
            ["xsel", "--clipboard", "--input"],
            input=text.encode(), check=True,
        )
    elif shutil.which("wl-copy"):
        subprocess.run(["wl-copy", text], check=True)
    else:
        raise RuntimeError("No clipboard tool found (install xclip, xsel, or wl-copy)")


def mask_api_key(key: str) -> str:
    """Partially mask an API key, showing first 6 and last 4 chars."""
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:6]}...{key[-4:]}"


# ── Status formatting ───────────────────────────────────────────────

STATUS_MAP = {
    "ok": "ok",
    "pending": "~",
    "no_access": "no",
    "unsupported": "unsup",
    "timeout": "timeout",
    "error": "err",
    "rate_limited": "rate",
}

STATUS_COLORS = {
    "ok": "[green]ok[/]",
    "pending": "[dim italic]~[/]",
    "no_access": "[red]no[/]",
    "unsupported": "[dim]unsup[/]",
    "timeout": "[yellow]timeout[/]",
    "error": "[red bold]err[/]",
    "rate_limited": "[dark_orange]rate[/]",
}


def format_status(status: str) -> str:
    """Return a Rich-markup colored status label."""
    return STATUS_COLORS.get(status, f"[dim]{STATUS_MAP.get(status, '?')}[/]")


def plain_status(status: str) -> str:
    """Return a plain-text status label (for CSV export)."""
    return STATUS_MAP.get(status, "?")


# ── Model Detail Modal ──────────────────────────────────────────────


class ModelDetailScreen(ModalScreen):
    """Modal screen showing model details with timing and API key."""

    BINDINGS = [("escape", "dismiss", "Close")]

    def __init__(self, model: ModelInfo, api_key: str):
        self.model = model
        self.api_key = api_key
        super().__init__()

    def compose(self) -> ComposeResult:
        m = self.model
        size_s = f"{m.size_b:.0f}B" if m.size_b is not None else "-"
        ttft_s = f"{m.ttft_ms:.0f} ms" if m.ttft_ms is not None else "-"
        total_s = f"{m.total_time_ms:.0f} ms" if m.total_time_ms is not None else "-"
        latency_s = f"{m.latency_ms:.0f} ms" if m.latency_ms is not None else "-"
        prefill_s = f"{m.prefill_ms:.0f} ms" if m.prefill_ms is not None else "-"

        with Static(id="model-detail"):
            yield Static(f"[bold]Model:[/bold] {m.id}", id="model-id")
            yield Static(f"[bold]API Provider:[/bold] {m.api_provider}", id="provider")
            yield Static(f"[bold]Model Provider:[/bold] {m.model_provider}", id="model-provider")
            yield Static(f"[bold]Type:[/bold] {m.model_type}", id="model-type")
            yield Static(f"[bold]Size:[/bold] {size_s}", id="model-size")
            yield Static(f"[bold]Status:[/bold] {format_status(m.status)}", id="model-status")
            yield Static(f"[bold]Latency:[/bold] {latency_s}", id="model-latency")
            yield Static(f"[bold]TTFT:[/bold] {ttft_s}", id="model-ttft")
            yield Static(f"[bold]Total Time:[/bold] {total_s}", id="model-total")
            yield Static(f"[bold]Prefill:[/bold] {prefill_s}", id="model-prefill")
            yield Static(f"[bold]API Key:[/bold] {mask_api_key(self.api_key)}", id="api-key")
            yield Button("Copy Model ID", id="copy-model-btn", variant="default")
            yield Button("Copy API Key", id="copy-btn", variant="primary")
            yield Button("Close", id="close-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "copy-model-btn":
            try:
                copy_to_clipboard(self.model.id)
                self.app.notify("Model ID copied to clipboard")
            except RuntimeError as e:
                self.app.notify(str(e), severity="error")
        elif event.button.id == "copy-btn":
            try:
                copy_to_clipboard(self.api_key)
                self.app.notify("API Key copied to clipboard")
            except RuntimeError as e:
                self.app.notify(str(e), severity="error")
        elif event.button.id == "close-btn":
            self.dismiss()


# ── Main Application ────────────────────────────────────────────────


class LLMPing(App):
    TITLE = "LLMPing"
    CSS = """
    Screen { background: $surface; }
    DataTable { height: 1fr; margin: 0 1; }
    #status-bar {
        height: 1; padding: 0 1; background: $panel;
        color: $text; content-align: center middle;
    }
    #search-bar {
        display: none;
        height: 3; padding: 0 1; margin: 0 1;
    }
    #search-bar.visible { display: block; }
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
        Binding("s", "sort_by_latency", "Sort: Latency"),
        Binding("p", "sort_by_api", "Sort: API"),
        Binding("m", "sort_by_model_provider", "Sort: Model"),
        Binding("t", "sort_by_ttft", "Sort: TTFT"),
        Binding("z", "sort_by_size", "Sort: Size"),
        Binding("c", "sort_by_type", "Sort: Type"),
        Binding("f", "toggle_filter", "Filter"),
        Binding("slash", "search", "Search"),
        Binding("e", "export", "Export CSV"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="Search models... (Esc to clear)", id="search-bar")
        yield DataTable()
        yield Static(id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns(
            "Model", "API Provider", "Model Provider", "Status",
            "Latency (ms)", "TTFT (ms)", "Size (B)", "Type",
        )
        self.providers: list[BaseProvider] = []
        self.models: list[ModelInfo] = []
        self.show_all = False
        self._current_sort: str = "none"
        self._search_query: str = ""
        self.action_refresh()

    # ── Provider init ───────────────────────────────────────────────

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

    # ── Core check worker ───────────────────────────────────────────

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

        # Fetch models — resilient: skip failed providers instead of aborting
        all_models: list[ModelInfo] = []
        failed_providers: list[str] = []
        for prov in self.providers:
            try:
                ms = await prov.get_models()
                all_models.extend(ms)
            except Exception as e:
                failed_providers.append(prov.name)
                self.notify(f"⚠ {prov.name}: {e}", severity="warning")
                continue  # don't abort, continue with other providers

        if failed_providers and not all_models:
            status_widget.update(
                f"All providers failed: {', '.join(failed_providers)}"
            )
            return

        self.models = all_models
        total = len(self.models)

        table.clear()
        for m in self.models:
            size_s = f"{m.size_b:.0f}" if m.size_b is not None else "-"
            table.add_row(
                m.id, m.api_provider, m.model_provider,
                format_status("pending"), "-", "-", size_s, m.model_type,
                key=self._key(m),
            )

        status_widget.update(f"Checking latency for {total} models across {len(self.providers)} providers ...")

        # Per-provider semaphores for fairness
        provider_sems = {p.name: asyncio.Semaphore(4) for p in self.providers}
        completed = 0
        ok_count = 0
        start_time = time.monotonic()
        provider_stats: dict[str, dict[str, int]] = {
            p.name: {"ok": 0, "total": 0} for p in self.providers
        }

        async def check_model(m: ModelInfo) -> None:
            nonlocal completed, ok_count
            sem = provider_sems.get(m.api_provider)
            if sem is None:
                return
            async with sem:
                for prov in self.providers:
                    if prov.name == m.api_provider:
                        await prov.check_latency(m)
                        break
                completed += 1
                provider_stats[m.api_provider]["total"] += 1
                if m.status == "ok":
                    ok_count += 1
                    provider_stats[m.api_provider]["ok"] += 1
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

        # Provider health summary
        elapsed = time.monotonic() - start_time
        parts = []
        for prov in self.providers:
            stats = provider_stats[prov.name]
            parts.append(f"{prov.name}: {stats['ok']}/{stats['total']}")
        summary = " | ".join(parts)
        status_widget.update(
            f"Done in {elapsed:.1f}s  |  {summary}  |  "
            "Press [b]r[/] refresh, [b]s[/] sort, [b]f[/] filter, [b]/[/] search, [b]e[/] export"
        )

    # ── Row helpers ─────────────────────────────────────────────────

    @staticmethod
    def _key(m: ModelInfo) -> str:
        return f"{m.api_provider}::{m.id}"

    def _update_row(self, m: ModelInfo) -> None:
        table = self.query_one(DataTable)
        latency_s = f"{m.latency_ms:.0f}" if m.latency_ms is not None else "-"
        ttft_s = f"{m.ttft_ms:.0f}" if m.ttft_ms is not None else "-"
        try:
            table.update_cell(self._key(m), "Status", format_status(m.status))
            table.update_cell(self._key(m), "Latency (ms)", latency_s)
            table.update_cell(self._key(m), "TTFT (ms)", ttft_s)
        except KeyError:
            pass  # row may not exist if table was cleared during refresh

    # ── Filtering / sorting engine ──────────────────────────────────

    def _get_visible_models(self) -> list[ModelInfo]:
        """Return models filtered by accessibility and search query."""
        models = self.models
        if not self.show_all:
            models = [m for m in models if m.status == "ok"]
        if self._search_query:
            q = self._search_query.lower()
            models = [
                m for m in models
                if q in m.id.lower()
                or q in m.api_provider.lower()
                or q in m.model_provider.lower()
                or q in m.model_type.lower()
            ]
        return models

    def _apply_sort(self, models: list[ModelInfo]) -> list[ModelInfo]:
        """Sort a model list according to the current sort mode."""
        s = self._current_sort
        if s == "latency":
            with_val = [m for m in models if m.latency_ms is not None]
            without = [m for m in models if m.latency_ms is None]
            with_val.sort(key=lambda x: x.latency_ms)
            return with_val + without
        elif s == "api":
            return sorted(models, key=lambda x: (x.api_provider, x.model_provider, x.id))
        elif s == "model_provider":
            return sorted(models, key=lambda x: (x.model_provider, x.api_provider, x.id))
        elif s == "ttft":
            with_val = [m for m in models if m.ttft_ms is not None]
            without = [m for m in models if m.ttft_ms is None]
            with_val.sort(key=lambda x: x.ttft_ms)
            return with_val + without
        elif s == "size":
            with_val = [m for m in models if m.size_b is not None]
            without = [m for m in models if m.size_b is None]
            with_val.sort(key=lambda x: x.size_b)
            return with_val + without
        elif s == "type":
            return sorted(models, key=lambda x: (x.model_type, x.size_b or 999, x.id))
        return models  # "none" — insertion order

    def _sort_and_display(self, status_msg: Optional[str] = None) -> None:
        """Apply current filter + sort and repopulate the table."""
        models = self._get_visible_models()
        models = self._apply_sort(models)
        self._repopulate(models)
        if status_msg:
            self.query_one("#status-bar", Static).update(status_msg)

    def _repopulate(self, models: list[ModelInfo]) -> None:
        table = self.query_one(DataTable)
        table.clear()
        for m in models:
            latency_s = f"{m.latency_ms:.0f}" if m.latency_ms is not None else "-"
            ttft_s = f"{m.ttft_ms:.0f}" if m.ttft_ms is not None else "-"
            size_s = f"{m.size_b:.0f}" if m.size_b is not None else "-"
            table.add_row(
                m.id, m.api_provider, m.model_provider,
                format_status(m.status), latency_s, ttft_s, size_s, m.model_type,
                key=self._key(m),
            )

    # ── Worker error handling ───────────────────────────────────────

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state == WorkerState.ERROR and event.worker.error:
            self.query_one("#status-bar", Static).update(
                f"Error: {event.worker.error}"
            )

    # ── Actions ─────────────────────────────────────────────────────

    def action_refresh(self) -> None:
        self.models = []
        self._current_sort = "none"
        self._search_query = ""
        search_bar = self.query_one("#search-bar", Input)
        search_bar.value = ""
        search_bar.remove_class("visible")
        self.query_one(DataTable).clear()
        self.run_checks()

    def action_sort_by_latency(self) -> None:
        self._current_sort = "latency"
        visible = self._get_visible_models()
        visible = self._apply_sort(visible)
        self._repopulate(visible)
        self.query_one("#status-bar", Static).update(
            f"Showing {len(visible)} models sorted by latency"
        )

    def action_sort_by_api(self) -> None:
        self._current_sort = "api"
        visible = self._get_visible_models()
        visible = self._apply_sort(visible)
        self._repopulate(visible)
        self.query_one("#status-bar", Static).update(
            f"Showing {len(visible)} models sorted by API provider"
        )

    def action_sort_by_model_provider(self) -> None:
        self._current_sort = "model_provider"
        visible = self._get_visible_models()
        visible = self._apply_sort(visible)
        self._repopulate(visible)
        self.query_one("#status-bar", Static).update(
            f"Showing {len(visible)} models sorted by model provider"
        )

    def action_sort_by_ttft(self) -> None:
        self._current_sort = "ttft"
        visible = self._get_visible_models()
        visible = self._apply_sort(visible)
        self._repopulate(visible)
        self.query_one("#status-bar", Static).update(
            f"Showing {len(visible)} models sorted by TTFT"
        )

    def action_sort_by_size(self) -> None:
        self._current_sort = "size"
        visible = self._get_visible_models()
        visible = self._apply_sort(visible)
        self._repopulate(visible)
        self.query_one("#status-bar", Static).update(
            f"Showing {len(visible)} models sorted by size"
        )

    def action_sort_by_type(self) -> None:
        self._current_sort = "type"
        visible = self._get_visible_models()
        visible = self._apply_sort(visible)
        self._repopulate(visible)
        self.query_one("#status-bar", Static).update(
            f"Showing {len(visible)} models sorted by type"
        )

    def action_toggle_filter(self) -> None:
        self.show_all = not self.show_all
        visible = self._get_visible_models()
        visible = self._apply_sort(visible)
        self._repopulate(visible)
        if self.show_all:
            self.query_one("#status-bar", Static).update(
                f"Showing all {len(visible)} models (including inaccessible)"
            )
        else:
            self.query_one("#status-bar", Static).update(
                f"Showing {len(visible)} accessible models only"
            )

    # ── Search ──────────────────────────────────────────────────────

    def action_search(self) -> None:
        search_bar = self.query_one("#search-bar", Input)
        search_bar.add_class("visible")
        search_bar.focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search-bar":
            self._search_query = event.value
            self._sort_and_display()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search-bar":
            # Dismiss search bar, keep filter active
            event.input.remove_class("visible")
            self.query_one(DataTable).focus()

    def key_escape(self) -> None:
        search_bar = self.query_one("#search-bar", Input)
        if search_bar.has_class("visible"):
            search_bar.value = ""
            self._search_query = ""
            search_bar.remove_class("visible")
            self.query_one(DataTable).focus()
            self._sort_and_display()

    # ── Export ──────────────────────────────────────────────────────

    def action_export(self) -> None:
        visible = self._get_visible_models()
        visible = self._apply_sort(visible)
        if not visible:
            self.notify("Nothing to export", severity="warning")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llmping_results_{timestamp}.csv"
        try:
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Model", "API Provider", "Model Provider", "Status",
                    "Latency (ms)", "TTFT (ms)", "Size (B)", "Type",
                ])
                for m in visible:
                    writer.writerow([
                        m.id,
                        m.api_provider,
                        m.model_provider,
                        plain_status(m.status),
                        f"{m.latency_ms:.0f}" if m.latency_ms is not None else "",
                        f"{m.ttft_ms:.0f}" if m.ttft_ms is not None else "",
                        f"{m.size_b:.0f}" if m.size_b is not None else "",
                        m.model_type,
                    ])
            self.notify(f"Exported {len(visible)} models to {filename}")
        except OSError as e:
            self.notify(f"Export failed: {e}", severity="error")

    # ── Row interaction ─────────────────────────────────────────────

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key:
            key = str(event.row_key.value)
            for m in self.models:
                if self._key(m) == key:
                    size_s = f"{m.size_b:.0f}B" if m.size_b is not None else "-"
                    self.query_one("#status-bar", Static).update(
                        f"{m.id}  |  API: {m.api_provider}  |  "
                        f"Provider: {m.model_provider}  |  "
                        f"Type: {m.model_type}  |  "
                        f"Size: {size_s}  |  "
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
