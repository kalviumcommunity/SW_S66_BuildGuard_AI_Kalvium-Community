"""Run a multi-turn staff-support conversation from the command line."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, cast

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessageParam

from api_client import get_chat_client, get_chat_model
from conversation import ConversationHistory

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
    "What is the safest response when the available guidance does not answer my question, and how can I explain the evidence limitation, suggest a useful next step, avoid inventing a policy, and remain professional and concise?",
    "How should an employee prepare before contacting a manager about a workplace process, including the relevant dates, previous messages, current status, specific question, desired outcome, and any supporting records that can be shared appropriately?",
    "What should a staff member do after noticing that a workplace form is missing a required field, contains unclear instructions, or appears to use an older process, and how can they request correction without guessing what belongs in the field?",
    "How can someone distinguish between a routine workplace question and an urgent safety concern, what observable facts should they record, and how should they seek immediate help when the available procedure does not state an escalation route?",
    "If a request involves another team, how should the employee identify the handoff, summarize what has already happened, provide the relevant reference details, and avoid promising a response time that is not documented?",
    "What is a professional way to ask whether a workplace procedure has changed, including how to refer to the version being used, identify the uncertain section, request the current source, and communicate the impact of the uncertainty?",
    "How should an employee respond when a colleague offers informal advice that differs from the available official guidance, and what should be checked before treating either explanation as an authoritative instruction?",
    "What details are useful in a concise workplace follow-up message when a request has been submitted, the expected next step is unclear, and the employee needs help without implying that silence proves approval or rejection?",
    "How should sensitive workplace information be handled in a request for procedural clarification, including what is necessary to share, what should be omitted, and how to ask for an appropriate confidential channel when one is not specified?",
    "What should a staff-support assistant say when asked to name a contact, deadline, approval authority, or exception that is not present in the system prompt or earlier conversation, and why is a confident guess unsafe?",
    "H  ow can a workplace response remain useful and concise when the available evidence is incomplete, by summarizing what is known, stating what cannot be verified, suggesting a safe verification step, and avoiding invented procedural details?",
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
        history = ConversationHistory(load_system_prompt())
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
            response = client.chat.completions.create(
                model=model,
                messages=cast(list[ChatCompletionMessageParam], history.messages),
            )
            logger.info("Incoming response payload: %s", response_payload(response))

            content = response.choices[0].message.content or ""
            history.add_message("assistant", content)
            print("LLM request: SUCCESS")
            print(f"Assistant: {content}")
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
    except (IndexError, KeyError, TypeError) as error:
        logger.error("Unexpected response format: %s", error)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
