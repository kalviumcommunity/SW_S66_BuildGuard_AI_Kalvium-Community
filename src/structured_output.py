"""A small structured-output flow for the staff-support assistant."""

from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from api_client import get_chat_client, get_chat_model
from prompt_templates import render_staff_question

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "staff_assistant_system.txt"


class AssistantResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    source: str


class StructuredOutputError(RuntimeError):
    """Raised when a response is incomplete, refused, malformed, or invalid."""


def load_system_prompt() -> str:
    """Load the existing staff-assistant system prompt."""
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _refusal(response: Any) -> str | None:
    """Return a refusal found on a Responses API output item, if any."""
    for item in _value(response, "output", []) or []:
        for content in _value(item, "content", []) or []:
            if _value(content, "type") == "refusal":
                return str(_value(content, "refusal", "The model refused the request."))
    return None


def parse_response(response: Any) -> AssistantResponse:
    """Check a Responses API result and validate its JSON output."""
    if _value(response, "status") == "incomplete":
        reason = _value(_value(response, "incomplete_details", {}), "reason")
        if reason in {"max_output_tokens", "content_filter"}:
            raise StructuredOutputError(f"Response incomplete: {reason}")
        raise StructuredOutputError(f"Response incomplete: {reason or 'unknown reason'}")

    refusal = _refusal(response)
    if refusal:
        raise StructuredOutputError(f"Response refused: {refusal}")

    return parse_json(_value(response, "output_text", ""))


def parse_json(output_text: str) -> AssistantResponse:
    """Decode and validate one structured JSON response."""
    try:
        data = json.loads(output_text)
    except JSONDecodeError as error:
        raise StructuredOutputError(f"Malformed JSON: {error.msg}") from error

    try:
        return AssistantResponse.model_validate(data)
    except ValidationError as error:
        raise StructuredOutputError(f"Invalid AssistantResponse: {error}") from error


def request_assistant_response(
    question: str,
    *,
    system_prompt: str | None = None,
    history: list[dict[str, str]] | None = None,
    client: Any | None = None,
    model: str | None = None,
) -> AssistantResponse:
    """Request and validate one structured assistant response."""
    effective_system_prompt = system_prompt or load_system_prompt()
    rendered_prompt = render_staff_question(
        context=effective_system_prompt,
        question=question,
    )
    messages = [
        dict(message)
        for message in (history or [])
        if message.get("role") != "system"
    ]
    if messages:
        messages[-1]["content"] = rendered_prompt
    else:
        messages = [{"role": "user", "content": rendered_prompt}]

    response = (client or get_chat_client()).responses.create(
        model=model or get_chat_model(),
        instructions=effective_system_prompt,
        input=cast(Any, messages),
        text={"format": {"type": "json_object"}},
    )
    return parse_response(response)


def demonstrate_local_cases() -> None:
    """Demonstrate the four expected local parsing outcomes."""
    cases = {
        "malformed JSON fails": '{"answer":"Incomplete"',
        "valid JSON succeeds/recovery": '{"answer":"Ask your manager.","source":"Staff handbook"}',
        "missing source fails": '{"answer":"Ask your manager."}',
        "wrong answer type fails": '{"answer":42,"source":"Staff handbook"}',
    }
    for name, output_text in cases.items():
        try:
            result = parse_json(output_text)
        except StructuredOutputError as error:
            print(f"{name}: {error}")
        else:
            print(f"{name}: {result.model_dump_json()}")
