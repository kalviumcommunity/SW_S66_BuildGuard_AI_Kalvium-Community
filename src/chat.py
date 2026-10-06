"""Run a multi-turn staff-support conversation from the command line."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)

from api_client import get_chat_client, get_chat_model
from conversation import ConversationHistory
from structured_output import (
    StructuredOutputError,
    request_assistant_response,
)

logger = logging.getLogger("conversation_chat")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "staff_assistant_system.txt"
LOG_PATH = PROJECT_ROOT / "outputs" / "conversation_history.log"

MAX_TOKENS = 1000
SAFETY_MARGIN = 100
TARGET_TOKENS = MAX_TOKENS - SAFETY_MARGIN


QUESTIONS = [
    "How do I request leave when I need several consecutive working days away from the office, what information should I include in the request, who should review it, and what should I do if the planned dates are urgent or overlap with an important team responsibility?",
    "Where should I submit a leave request for manager review, how can I confirm that it was received, what details should I retain for my records, and how should I follow up if the responsible person is unavailable or does not respond?",
    "Who should I contact about a workplace issue or safety concern involving damaged equipment, an unsafe room, a blocked exit, or another possible hazard, and which factual details would help the organization investigate without assigning blame?",
    "What should I do if my request is rejected, how should I ask for an explanation, which parts of the decision should I document, and how can I request clarification without assuming that an appeal process, deadline, or exception exists?",
    "How should I follow up if I have not received an answer to my workplace request, what dates and earlier messages should I include, how should I keep the reminder professional, and what should I avoid claiming when no response deadline is confirmed?",
    "What should I do when two workplace procedure documents appear to conflict about an approval, reporting route, required form, or expected timing, and how can I ask the procedure owner to clarify the difference?",
    "How can I report a workplace concern while keeping the report factual, separating observations from assumptions, protecting unnecessary personal information, recording the relevant time and location, and identifying the safest next step when the urgency is uncertain?",
    "What information should I provide when asking for clarification about a procedure, including the current situation, relevant dates, people or teams involved, supporting records, decisions already made, and the specific outcome I am requesting?",
    "How should I handle a workplace question involving an unconfirmed deadline, contact person, approval requirement, exception, or consequence when the available conversation does not provide enough evidence to verify the policy?",

]


def configure_logging() -> None:
    """Write request, response, and error details to the task log file."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def load_system_prompt() -> str:
    """Read the system prompt from the project file."""
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def response_payload(response: Any) -> str:
    """Serialize an SDK response for diagnostic logging."""
    if hasattr(response, "model_dump"):
        payload = response.model_dump(mode="json")
    elif hasattr(response, "to_dict"):
        payload = response.to_dict()
    else:
        payload = response
    return json.dumps(payload, ensure_ascii=False, default=str)



def main() -> int:
    configure_logging()
    logger.info("Starting conversation history demo")
    try:
        system_prompt = load_system_prompt()
        history = ConversationHistory(system_prompt)
        client = get_chat_client()
        model = get_chat_model()

        print(
            f"Token budget: {MAX_TOKENS} "
            f"(effective target: {TARGET_TOKENS})"
        )

        for turn, question in enumerate(QUESTIONS, start=1):
            print("\n--------------------------------")
            print(f"Turn {turn}")
            print("--------------------------------")
            print(f"Question: {question}")
            logger.info("Turn %s question: %s", turn, question)

            history.add_message("user", question)
            tokens_before_trim = history.count_tokens()
            print(f"History tokens: {tokens_before_trim}")
            logger.info("Turn %s history tokens before trimming: %s", turn, tokens_before_trim)

            if tokens_before_trim > TARGET_TOKENS:
                print("Action: Trimming oldest conversation turn")
                removed_pairs = history.trim_history(TARGET_TOKENS)
                print(f"Removed turns: {removed_pairs}")
                logger.info("Turn %s trimmed pairs: %s", turn, removed_pairs)
            else:
                print("Action: No trimming required")
                logger.info("Turn %s required no trimming", turn)

            request_tokens = history.count_tokens()
            print(f"History tokens after trimming: {request_tokens}")
            logger.info("Turn %s request tokens: %s", turn, request_tokens)
            print("LLM request: sending")

            logger.info(
                "Outgoing messages: %s",
                json.dumps(history.messages, ensure_ascii=False),
            )
            result = request_assistant_response(
                question,
                system_prompt=system_prompt,
                history=history.messages,
                client=client,
                model=model,
            )
            logger.info("Validated response payload: %s", result.model_dump_json())

            assistant_content = result.model_dump_json()
            history.add_message("assistant", assistant_content)
            print("LLM request: SUCCESS")
            print(assistant_content)
            logger.info("Turn %s LLM request succeeded", turn)

    except ValueError as error:
        logger.error("Configuration error: %s", error)
        return 1
    except AuthenticationError:
        logger.error("Authentication failed (HTTP 401). Check API_KEY.")
        return 1
    except RateLimitError:
        logger.error("Rate limit exceeded (HTTP 429). Please try again later.")
        return 1
    except APIConnectionError:
        logger.error("Could not connect to the API. Check API_BASE_URL.")
        return 1
    except APIStatusError as error:
        logger.error("API request failed (HTTP %s): %s", error.status_code, error.message)
        return 1
    except APIError as error:
        logger.error("API error: %s", error)
        return 1
    except StructuredOutputError as error:
        logger.error("Structured response validation failed: %s", error)
        return 1
    except (IndexError, KeyError, TypeError) as error:
        logger.error("Unexpected response format: %s", error)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
