"""Count project text tokens and estimate input/output model cost."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tiktoken


ENCODING_NAME = "cl100k_base"
# Example rates in USD per one million tokens; replace with the target model's rates.
INPUT_PRICE_PER_MILLION = 0.15
OUTPUT_PRICE_PER_MILLION = 0.60
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Sample:
    name: str
    input_text: str
    output_text: str


def token_count(encoding: tiktoken.Encoding, text: str) -> int:
    return len(encoding.encode(text))


def money_for_tokens(tokens: int, price_per_million: float) -> float:
    return tokens / 1_000_000 * price_per_million


def build_samples() -> list[Sample]:
    prompt_text = (PROJECT_ROOT / "prompts" / "staff_assistant_system.txt").read_text(
        encoding="utf-8"
    )
    readme_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    return [
        Sample(
            "short question",
            "What is the leave approval process?",
            "Check the available staff guidance for the approval steps.",
        ),
        Sample(
            "prompt paragraph",
            prompt_text,
            "I don't have enough information to answer that reliably.",
        ),
        Sample(
            "full project README",
            readme_text,
            "The project uses uv, shared API configuration, prompt comparison, and token analysis.",
        ),
    ]


def print_report(samples: list[Sample], encoding: tiktoken.Encoding) -> None:
    total_input = 0
    total_output = 0
    print(f"Tokenizer: {ENCODING_NAME}")
    print(f"Input rate: ${INPUT_PRICE_PER_MILLION:.2f} per 1M tokens")
    print(f"Output rate: ${OUTPUT_PRICE_PER_MILLION:.2f} per 1M tokens")
    print()

    for sample in samples:
        input_tokens = token_count(encoding, sample.input_text)
        output_tokens = token_count(encoding, sample.output_text)
        total_input += input_tokens
        total_output += output_tokens
        chars = len(sample.input_text)
        words = len(sample.input_text.split())
        print(f"[{sample.name}]")
        print(f"  input characters: {chars}")
        print(f"  input words: {words}")
        print(f"  input tokens: {input_tokens}")
        print(f"  output tokens: {output_tokens}")
        print(f"  tokens per character: {input_tokens / max(chars, 1):.4f}")
        print()

    input_cost = money_for_tokens(total_input, INPUT_PRICE_PER_MILLION)
    output_cost = money_for_tokens(total_output, OUTPUT_PRICE_PER_MILLION)
    print("[total estimate]")
    print(f"  input tokens: {total_input}")
    print(f"  output tokens: {total_output}")
    print(f"  estimated input cost: ${input_cost:.8f}")
    print(f"  estimated output cost: ${output_cost:.8f}")
    print(f"  estimated total cost: ${input_cost + output_cost:.8f}")
    print()
    print("[length-token relationship]")
    print("Character and token counts generally increase together, but they are not")
    print("proportional: whitespace, punctuation, long words, and code can tokenize")
    print("differently even when strings have similar character lengths.")


def main() -> int:
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    print_report(build_samples(), encoding)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
