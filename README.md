# LLMPing

A TUI tool that pings free LLM models across multiple API providers and shows you latency at a glance.

Think `ping` for large language models - see which models are responsive, which are not, and how fast they answer - all in one sortable table.

## Features

- Checks latency for **free-tier models** across 4 providers simultaneously
- Sorts by latency, API provider, model provider, or TTFT
- Filters to show only accessible models
- Auto-detects which API keys you have configured (no unused provider clutter)
- Concurrent requests with live progress

## Supported Providers

| Provider | Free Models | Auth Env Var |
|---|---|---|
| [NVIDIA NIM](https://build.nvidia.com) | ~130 | `NVIDIA_API_KEY` |
| [OpenRouter](https://openrouter.ai) | ~28 | `OPENROUTER_API_KEY` |
| [Groq](https://groq.com) | ~16 | `GROQ_API_KEY` |
| [Z.AI](https://z.ai) | 2 (GLM-4.5-Flash, GLM-4.7-Flash) | `ZAI_API_KEY` |

Only providers with a non-empty API key in your `.env` file will be activated - empty or missing keys are silently skipped.

## Setup

### 1. Get API Keys (free)

- **NVIDIA**: Sign up at [build.nvidia.com](https://build.nvidia.com) → Generate API Key
- **OpenRouter**: Sign up at [openrouter.ai/keys](https://openrouter.ai/keys) → Create key
- **Groq**: Sign up at [console.groq.com/keys](https://console.groq.com/keys) → Create API Key
- **Z.AI**: Sign up at [z.ai](https://z.ai) → API Keys → Create key (free models don't need a balance)

All of these are genuinely free - no credit card needed for the free tier.

### 2. Clone and Setup

```bash
git clone <repo-url> llmping
cd llmping

python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Configure API Keys

Copy the example file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env` and add the keys you want to use. Leave any line empty to skip that provider:

```
NVIDIA_API_KEY=nvapi-abc123...
GROQ_API_KEY=gsk_abc123...
ZAI_API_KEY=abc123...
OPENROUTER_API_KEY=sk-or-abc123...
```

### 4. Run

```bash
python app.py
```

## Usage

| Key | Action |
|---|---|
| `r` | Refresh - re-fetch and re-check all models |
| `s` | Sort by latency (fastest first) |
| `p` | Sort by API provider |
| `m` | Sort by model provider |
| `t` | Sort by TTFT (time to first token) |
| `f` | Toggle filter - show all models or only accessible ones |
| `q` | Quit |
| ↑/↓ | Navigate rows |
| Enter | Select a row to see details |

Click a row or use arrow keys to see detailed info in the status bar (provider, TTFT, total time, status).

## Status Labels

| Label | Meaning |
|---|---|
| ok | Accessible - latency measured |
| no | No access (402/403 or insufficient balance) |
| unsup | Not supported by model |
| timeout | Timed out |
| rate | Rate limited |
| err | Error |
| ~ | Pending / checking |

## Privacy

**We are not stealing your API keys.** Promise. XD

Seriously though:

- The app runs entirely **on your machine** - no telemetry, no analytics, no external calls except to the LLM API providers themselves.
- API keys stay in your `.env` file (which is gitignored by default). They are only sent directly to the provider whose header they belong to.
- No data leaves your computer except the single "Hi" prompt sent to each model to measure latency.
- The source is ~350 lines of Python - fully auditable. If you're paranoid, read it.

## How It Works

1. LLMPing discovers free models from each provider's API.
2. For each model, it sends a minimal chat request (`"Hi"`, `max_tokens=5`) and measures wall-clock time.
3. Results are displayed in a live-updating TUI table.
4. NVIDIA models use server-reported timing (`nvext.timing`); all others use client-side timing.
5. Only the first response token (TTFT) and total request time are recorded - no generated content is stored.

## Project Structure

```
llmping/
├── app.py                  # Main TUI application
├── providers/
│   ├── __init__.py         # Base provider, ModelInfo, provider registry
│   ├── nvidia.py           # NVIDIA NIM provider
│   ├── openrouter.py       # OpenRouter provider
│   ├── groq.py             # Groq provider
│   └── zai.py              # Z.AI provider
├── .env                    # Your API keys (gitignored)
├── .env.example            # Key template
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── README.md               # This file
```

## License

MIT License - see [LICENSE](LICENSE).
