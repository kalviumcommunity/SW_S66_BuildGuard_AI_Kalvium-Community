# BuildGuard AI

AI project setup with [uv](https://docs.astral.sh/uv/) as the Python package manager.

## Workspace layout

| Path        | Purpose                              |
| ----------- | ------------------------------------ |
| `data/`     | Input documents (git-ignored)        |
| `src/`      | Source code                          |
| `prompts/`  | Prompt templates                     |
| `outputs/`  | Generated results; sample log is tracked |

## Setup

1. **Create the environment and install dependencies** (uses `pyproject.toml` + `uv.lock`):

   ```bash
   uv sync
   ```

2. **Configure your keys** — copy the template and fill in real values (`.env` is git-ignored):

   ```bash
   copy .env.example .env    # Windows
   cp .env.example .env      # macOS / Linux
   ```

3. **Run the chat completion client**:

   ```bash
   uv run python src/chat_completion.py
   ```

The client logs the request messages, response payload, and token usage without
logging the API key. Its assistant response is printed to standard output.
Configuration and API failures are reported without a raw traceback, and failed
requests return a nonzero exit code.

## Required keys

See `.env.example`:

- `API_BASE_URL` — API base URL
- `API_KEY` — API key
- `CHAT_MODEL` — chat model name
- `EMBEDDING_MODEL` — embedding model name (used by other project components, if configured)

## Sample output

A representative response is committed at `outputs/chat_completion_sample.log`.
It contains placeholder response metadata and no real credentials.

## Clean-run confirmation

Verified on a fresh setup: `.venv` was deleted and rebuilt from scratch with `uv sync`
(88 packages resolved from `uv.lock`, installed successfully), and `.env.example` copies
cleanly to `.env`. Re-run `uv sync` after cloning to reproduce this environment.
