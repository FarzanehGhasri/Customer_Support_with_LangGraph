"""
Response-composition abstraction.

An agent has two jobs: work out *what is true* (tools, retrieval) and *say it*.
This Protocol covers only the second. Splitting it out means:

* agents no longer depend on :class:`ChatModelProvider` at all -- they depend on
  "something that can phrase an answer";
* the whole graph runs with no API key by substituting a template composer,
  which is how the offline test suite exercises every node;
* the truth of an answer never depends on the composer, because the caller
  always supplies a ``fallback`` that is already correct.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ResponseComposer(Protocol):
    """Turns verified facts into customer-facing prose."""

    def compose(self, system_prompt: str, user_message: str, *, fallback: str) -> str:
        """Phrase a reply.

        Args:
            system_prompt: Instructions plus the verified facts.
            user_message: What the customer wrote.
            fallback: A correct-but-unpolished answer to return if composition
                is impossible. It must never be empty: an agent that goes silent
                looks to the guardrail like an agent that succeeded.
        """
        ...
