"""Send a single chat completion request to an OpenAI-compatible API."""

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


logger = logging.getLogger("chat_completion")

SYSTEM_MESSAGE = "You are a helpful assistant. Answer clearly and concisely."
USER_MESSAGE = "Explain in one sentence why software tests are valuable."


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")



def response_payload(response: Any) -> str:
    """Serialize an SDK response for readable diagnostic logging."""
    if hasattr(response, "model_dump"):
        payload = response.model_dump(mode="json")
    elif hasattr(response, "to_dict"):
        payload = response.to_dict()
    else:
        payload = response
    return json.dumps(payload, ensure_ascii=False, default=str)


def main() -> int:
    configure_logging()
    try:
        model = get_chat_model()
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": USER_MESSAGE},
        ]
        logger.info("Outgoing messages: %s", json.dumps(messages, ensure_ascii=False))

        client = get_chat_client()
        response = client.chat.completions.create(model=model, messages=messages)

        logger.info("Incoming response payload: %s", response_payload(response))
        if response.usage is not None:
            logger.info("Token usage: %s", response.usage)

        content = response.choices[0].message.content
        if content:
            print(content)
        return 0
    except ValueError as error:
        logger.error("Configuration error: %s", error)
    except AuthenticationError:
        logger.error("Authentication failed (HTTP 401). Check API_KEY and its permissions.")
    except RateLimitError:
        logger.error("Rate limit exceeded (HTTP 429). Please wait and try again.")
    except APIConnectionError:
        logger.error("Could not connect to the API. Check API_BASE_URL and network access.")
    except APIStatusError as error:
        logger.error("API request failed (HTTP %s): %s", error.status_code, error.message)
    except APIError as error:
        logger.error("API error: %s", error)
    except (IndexError, KeyError, TypeError) as error:
        logger.error("Unexpected response format: %s", error)
    return 1


if __name__ == "__main__":
    sys.exit(main())
