"""
Domain enumerations.

Why enums instead of bare strings?
---------------------------------
The assignment spec stores ``department``, ``sentiment`` and ``next_step`` inside
the graph state as plain strings.  We keep that wire format (so the state stays
exactly as the spec describes), but every *producer* and *consumer* of those
strings goes through these enums.  That gives us:

* a single source of truth for the allowed values (no typo-driven routing bugs);
* automatic JSON-schema ``enum`` constraints when the enums are used inside the
  Pydantic models that drive ``with_structured_output``;
* Open/Closed friendliness -- adding a new department later means adding one
  member here plus one specialist class, and *not* editing the router logic.

All enums subclass ``str`` so that ``Department.BILLING == "Billing"`` is True and
the members serialise straight into the ``TypedDict`` state without conversion.
"""

from __future__ import annotations

from enum import Enum


class Department(str, Enum):
    """The queues a user request can be routed to.

    ``TRIAGE`` is not a real department: it is the "back to reception" value a
    specialist returns when it receives a request that is not its business
    (the spec requires the Billing agent to bounce technical questions back).
    """

    BILLING = "Billing"
    TECHNICAL = "Technical"
    GENERAL = "General"
    TRIAGE = "Triage"

    @classmethod
    def specialists(cls) -> tuple["Department", ...]:
        """Departments that own a specialist node (i.e. can terminate a route)."""
        return (cls.BILLING, cls.TECHNICAL, cls.GENERAL)


class Sentiment(str, Enum):
    """Tone of the *user's* message, as judged by the sentiment guardrail."""

    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"

    @property
    def needs_human(self) -> bool:
        """Only an angry/negative user triggers the human-in-the-loop interrupt."""
        return self is Sentiment.NEGATIVE


class NextStep(str, Enum):
    """Control-flow variable written into ``SupportState['next_step']``.

    LangGraph conditional edges map these values to node names.  Keeping the
    routing vocabulary in one enum means the conditional-edge mapping can be
    generated from the enum instead of hand-written in several places.
    """

    TRIAGE = "triage"
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"
    GUARDRAIL = "guardrail"
    HUMAN_REVIEW = "human_review"
    FINISH = "finish"

    @classmethod
    def for_department(cls, department: "Department | str") -> "NextStep":
        """Map a department onto the node that serves it.

        Centralising this mapping is what lets the Triage node stay ignorant of
        node *names*: it only ever decides on a :class:`Department`.
        """
        department = Department(department)
        mapping = {
            Department.BILLING: cls.BILLING,
            Department.TECHNICAL: cls.TECHNICAL,
            Department.GENERAL: cls.GENERAL,
            Department.TRIAGE: cls.TRIAGE,
        }
        return mapping[department]
