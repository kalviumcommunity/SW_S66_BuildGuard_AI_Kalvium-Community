"""Compare vague and constrained prompts for the same staff question."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessageParam

from api_client import get_chat_client, get_chat_model
from prompt_templates import render_staff_question

logger = logging.getLogger("prompt_comparison")

USER_QUESTION = (
    "What should I do if I cannot find the information I need in the staff handbook?"
)
VAGUE_SYSTEM_PROMPT = "Answer the staff member's question helpfully."
CONSTRAINED_SYSTEM_PROMPT = (
    "You are a staff-support assistant. Answer questions about workplace procedures "
    "using only information provided in the conversation or retrieved context. Do "
    "not invent policies, deadlines, contacts, or facts. Respond in a professional, "
    "clear tone in no more than three short sentences. If the information is not "
    "available, say: 'I don't have enough information to answer that reliably.'"
)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def response_payload(response: Any) -> str:
    if hasattr(response, "model_dump"):
        payload = response.model_dump(mode="json")
    elif hasattr(response, "to_dict"):
        payload = response.to_dict()
    else:
        payload = response
    return json.dumps(payload, ensure_ascii=False, default=str)


def request_completion(client: Any, model: str, system_prompt: str) -> Any:
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": render_staff_question(
                context=system_prompt,
                question=USER_QUESTION,
            ),
        },
    ]
    logger.info("Outgoing messages: %s", json.dumps(messages, ensure_ascii=False))
    response = client.chat.completions.create(model=model, messages=messages)
    logger.info("Incoming response payload: %s", response_payload(response))
    if response.usage is not None:
        logger.info("Token usage: %s", response.usage)
    return response


def main() -> int:
    configure_logging()
    try:
        client = get_chat_client()
        model = get_chat_model()
        results = []
        for label, prompt in (
            ("Variation A: Vague prompt", VAGUE_SYSTEM_PROMPT),
            ("Variation B: Constrained prompt", CONSTRAINED_SYSTEM_PROMPT),
        ):
            response = request_completion(client, model, prompt)
            content = response.choices[0].message.content or ""
            results.append((label, content))

        for label, content in results:
            print(f"\n{label}\n{'-' * len(label)}\n{content}")
        return 0
    except ValueError as error:
        logger.error("Configuration error: %s", error)
    except AuthenticationError:
        logger.error(
            "Authentication failed (HTTP 401). Check API_KEY and its permissions."
        )
    except RateLimitError:
        logger.error("Rate limit exceeded (HTTP 429). Please wait and try again.")
    except APIConnectionError:
        logger.error(
            "Could not connect to the API. Check API_BASE_URL and network access."
        )
    except APIStatusError as error:
        logger.error(
            "API request failed (HTTP %s): %s", error.status_code, error.message
        )
    except APIError as error:
        logger.error("API error: %s", error)
    except (IndexError, KeyError, TypeError) as error:
        logger.error("Unexpected response format: %s", error)
    return 1


if __name__ == "__main__":
    sys.exit(main())
