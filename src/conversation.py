"""In-memory conversation history management."""

from __future__ import annotations

import tiktoken

from token_cost_estimate import ENCODING_NAME, token_count

_ENCODING = tiktoken.get_encoding(ENCODING_NAME)


class ConversationHistory:
    """Store a system prompt and user/assistant conversation turns."""

    def __init__(self, system_prompt: str) -> None:
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

    def add_message(self, role: str, content: str) -> None:
        """Append a user or assistant message to the history."""
        if role not in {"user", "assistant"}:
            raise ValueError("Only user and assistant messages may be added.")
        self.messages.append({"role": role, "content": content})

    def count_tokens(self) -> int:
        """Count tokens in the message content currently sent to the LLM."""
        return sum(
            token_count(_ENCODING, message["content"])
            for message in self.messages
        )

    def trim_history(self, max_tokens: int) -> int:
        """Remove oldest complete user/assistant pairs until within max_tokens."""
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")

        removed_pairs = 0
        while self.count_tokens() > max_tokens:
            pair_start = next(
                (
                    index
                    for index in range(1, len(self.messages) - 1)
                    if self.messages[index]["role"] == "user"
                    and self.messages[index + 1]["role"] == "assistant"
                ),
                None,
            )
            if pair_start is None:
                break

            del self.messages[pair_start : pair_start + 2]
            removed_pairs += 1

        return removed_pairs
