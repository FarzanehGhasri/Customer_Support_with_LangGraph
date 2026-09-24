"""
General agent -- the third destination of the triage classifier.

The assignment names GENERAL as a triage category but does not describe a node
for it, so this is deliberately the thinnest agent in the project: no tools, no
retrieval, nothing to get wrong. It exists so that greetings and off-topic
messages have somewhere to go, and so that the triage loop-guard has a terminal
destination.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from ..domain.enums import NextStep
from ..interfaces.composition import ResponseComposer
from ..prompts.specialists import GENERAL_ANSWER_PROMPT
from .base import BaseSupportNode

logger = logging.getLogger(__name__)


class GeneralAgent(BaseSupportNode):
    """Handles greetings, thanks and anything without a specialist."""

    node_name = "general"
    display_name = "Front Desk agent"

    def __init__(self, composer: ResponseComposer) -> None:
        self._composer = composer

    def handle(self, state) -> Mapping[str, Any]:
        message = self.current_message(state)
        draft = self.compose_reply(
            self._composer,
            state,
            GENERAL_ANSWER_PROMPT.format(user_message=message),
            message,
            fallback="Thanks for getting in touch! How can I help you today?",
        )

        return {
            "draft_response": draft,
            "next_step": NextStep.GUARDRAIL.value,
            "messages": [self.say(draft)],
        }
