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

4. **Compare prompt variations**:

   ```bash
   uv run python src/prompt_comparison.py
   ```

The prompt comparison sends the same staff question with a vague system prompt
and a clearer, constrained system prompt. It prints both responses and logs the
request messages, response payloads, and token usage without logging the API key.
Configuration and API failures are reported without a raw traceback, and failed
requests return a nonzero exit code.

## Required keys

See `.env.example`:

- `API_BASE_URL` — API base URL
- `API_KEY` — API key
- `CHAT_MODEL` — chat model name
- `EMBEDDING_MODEL` — embedding model name (used by other project components, if configured)

## Prompt comparison

The selected staff-support system prompt is stored in
`prompts/staff_assistant_system.txt`. It defines the assistant's role, limits
answers to available evidence, prevents invented policies, and specifies a
professional three-sentence limit with a safe fallback.

A representative comparison is committed at
`outputs/prompt_comparison_sample.log`. It includes the shared input, both
system prompts, illustrative outputs, token usage, and the selection rationale.
The original chat client sample remains at `outputs/chat_completion_sample.log`.

## Clean-run confirmation

Verified on a fresh setup: `.venv` was deleted and rebuilt from scratch with `uv sync`
(88 packages resolved from `uv.lock`, installed successfully), and `.env.example` copies
cleanly to `.env`. Re-run `uv sync` after cloning to reproduce this environment.
