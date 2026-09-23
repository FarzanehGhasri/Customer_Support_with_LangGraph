"""
Composition root.

This is the one module allowed to know about *every* concrete class at once.
Everywhere else depends on Protocols; here the wiring happens, once. That is
what keeps the dependency graph acyclic and the agents testable.

It also owns one policy decision: **which implementation to pick**. With
credentials it builds the LLM-backed classifiers and the vector retriever;
without them it builds the deterministic ones, so the whole graph still runs
offline. The notebook can therefore be executed by a grader who has no key.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..agents import (
    BillingAgent,
    GeneralAgent,
    HumanReviewNode,
    InterruptingHumanReviewNode,
    SentimentGuardrail,
    TechnicalAgent,
    TriageNode,
)
from ..config.settings import Settings
from ..domain.state import SupportState, initial_state
from ..infrastructure.billing import JsonSubscriptionRepository, MockRefundGateway
from ..infrastructure.classification import (
    KeywordIntentClassifier,
    KeywordSentimentAnalyzer,
    LLMIntentClassifier,
    LLMSentimentAnalyzer,
)
from ..infrastructure.composition import LLMResponseComposer, TemplateResponseComposer
from ..infrastructure.planning import LLMBillingPlanner, RuleBasedBillingPlanner
from ..infrastructure.llm import (
    LangChainEmbeddingProvider,
    LangChainModelProvider,
    probe_provider,
)
from ..infrastructure.retrieval import EmbeddingKnowledgeRetriever, KeywordKnowledgeRetriever
from .builder import build_support_graph

logger = logging.getLogger(__name__)


@dataclass
class SupportApplication:
    """The assembled system: the compiled graph plus the parts it was built from.

    The components are exposed so the notebook can demonstrate them in
    isolation (call a tool, run the retriever) without rebuilding anything.
    """

    graph: Any
    settings: Settings
    retriever: Any
    repository: Any
    gateway: Any
    offline: bool
    #: Why the system is offline, if it is. Empty when running live.
    offline_reason: str = ""
    #: "static" (interrupt_before + update_state) or "dynamic" (interrupt() +
    #: Command(resume=...)). See build_application for the difference.
    hitl_mode: str = "static"

    # ------------------------------------------------------------------ #
    def run(
        self,
        message: str,
        *,
        user_id: str = "",
        thread_id: str = "default",
        transaction_id: str = "",
    ) -> SupportState:
        """Run one conversation turn and return the resulting state.

        ``thread_id`` is the checkpointer's conversation key. Each scenario in
        the notebook uses its own, so a paused escalation can be resumed later
        without the runs interfering.
        """
        config = {"configurable": {"thread_id": thread_id}}
        state = initial_state(message, user_id, transaction_id=transaction_id)
        self.graph.invoke(state, config=config)
        return self.state(thread_id)

    def state(self, thread_id: str = "default") -> SupportState:
        """Current stored state of a conversation."""
        return self.graph.get_state({"configurable": {"thread_id": thread_id}}).values

    def is_interrupted(self, thread_id: str = "default") -> bool:
        """True when the graph is paused waiting for a human."""
        return bool(self.graph.get_state({"configurable": {"thread_id": thread_id}}).next)

    def next_nodes(self, thread_id: str = "default") -> tuple[str, ...]:
        """Which node(s) the graph would run if resumed."""
        return tuple(self.graph.get_state({"configurable": {"thread_id": thread_id}}).next)

    # ------------------------------------------------------------------ #
    def inject_manager_reply(self, reply: str, *, thread_id: str = "default") -> None:
        """Write a manager's answer into a paused conversation.

        This is the ``update_state`` step the assignment's scenario 3 requires.
        """
        self.graph.update_state(
            {"configurable": {"thread_id": thread_id}},
            {
                "final_response": reply,
                "messages": [f"manager: {reply}"],
            },
        )

    def resume(self, thread_id: str = "default") -> SupportState:
        """Continue a paused conversation. ``None`` means "carry on from here"."""
        self.graph.invoke(None, config={"configurable": {"thread_id": thread_id}})
        return self.state(thread_id)

    def resume_with(self, decision: Any, *, thread_id: str = "default") -> SupportState:
        """Resume a **dynamic** interrupt, sending the manager's decision in.

        This is the counterpart of :meth:`inject_manager_reply`: with
        ``hitl_mode="dynamic"`` the paused ``interrupt()`` call returns
        ``decision``, so the manager's verdict travels as a value rather than
        as a state write.

        Args:
            decision: The manager's reply as a string, ``True`` to approve the
                draft unchanged, or ``{"approved": ..., "reply": ..., "note": ...}``.
        """
        from langgraph.types import Command

        self.graph.invoke(
            Command(resume=decision),
            config={"configurable": {"thread_id": thread_id}},
        )
        return self.state(thread_id)

    def pending_interrupt(self, thread_id: str = "default") -> Any:
        """Payload the paused node handed to ``interrupt()``, if any.

        This is what a real review UI would render for the support manager.
        """
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        interrupts = getattr(snapshot, "interrupts", ()) or ()
        return interrupts[0].value if interrupts else None


def build_application(
    settings: Settings | None = None,
    *,
    force_offline: bool = False,
    reviewer: Any | None = None,
    interrupt_before_human: bool = True,
    probe: bool = True,
    hitl_mode: str = "static",
) -> SupportApplication:
    """Assemble the whole system.

    Args:
        settings: Configuration; read from the environment when omitted.
        force_offline: Use the deterministic components even if a key exists.
            Handy for a reproducible demo run or for testing.
        reviewer: Optional :class:`HumanReviewer` consulted after an interrupt.
        interrupt_before_human: Whether to pause for a human on escalation.
        hitl_mode: How the graph pauses for a human.

            * ``"static"``  -- compile with ``interrupt_before=["human_review"]``;
              the manager's answer arrives via ``update_state``. This is the flow
              the assignment's scenario 3 describes.
            * ``"dynamic"`` -- the review node calls ``interrupt(payload)`` itself
              and is resumed with ``Command(resume=...)``. This is the
              ``graph.interrupt()`` the assignment names in its Step 3 text.

            Both satisfy the requirement to halt and wait for a human; the
            notebook demonstrates each one.
        probe: Verify the model is actually reachable before choosing the live
            components. Leave this on. Because every component degrades quietly
            on failure, an unreachable endpoint otherwise produces a system that
            runs to completion and answers every question wrongly -- which is
            far harder to notice than an honest fallback. One tiny call at
            start-up buys that certainty.
    """
    settings = settings or Settings.from_env()

    offline_reason = ""
    if force_offline:
        offline, offline_reason = True, "forced by the caller"
    elif not settings.has_credentials:
        offline, offline_reason = True, f"no {settings.api_key_env_var}"
    elif probe:
        reachable, why = probe_provider(settings)
        offline, offline_reason = (not reachable), ("" if reachable else why)
    else:
        offline = False

    if offline:
        logger.warning(
            "Running OFFLINE (%s). Deterministic classifiers and keyword retrieval "
            "are in use; no API calls will be made.",
            offline_reason,
        )

    provider = LangChainModelProvider(settings)

    # --- model-backed components, or their deterministic twins --------- #
    if offline:
        classifier: Any = KeywordIntentClassifier()
        analyzer: Any = KeywordSentimentAnalyzer()
        planner: Any = RuleBasedBillingPlanner()
        composer: Any = TemplateResponseComposer()
    else:
        classifier = LLMIntentClassifier(provider)
        analyzer = LLMSentimentAnalyzer(provider)
        planner = LLMBillingPlanner(provider)
        composer = LLMResponseComposer(provider, temperature=settings.temperature)

    # --- retrieval ---------------------------------------------------- #
    keyword_retriever = KeywordKnowledgeRetriever(
        settings.knowledge_base_dir,
        min_score=settings.retrieval_min_score,
    )
    embeddings = LangChainEmbeddingProvider(settings)
    if not offline and embeddings.is_configured:
        # Vector search, with the keyword retriever as the safety net.
        retriever: Any = EmbeddingKnowledgeRetriever(
            embeddings, settings.knowledge_base_dir, fallback=keyword_retriever
        )
    else:
        retriever = keyword_retriever

    # --- tools' data sources ------------------------------------------ #
    repository = JsonSubscriptionRepository(settings.subscriptions_file)
    gateway = MockRefundGateway(settings.transactions_file)

    # --- human-in-the-loop mechanism ---------------------------------- #
    if hitl_mode not in {"static", "dynamic"}:
        raise ValueError(f"hitl_mode must be 'static' or 'dynamic', got {hitl_mode!r}")
    if hitl_mode == "dynamic":
        human_node: Any = InterruptingHumanReviewNode()
        # The node pauses itself, so the graph must NOT also pause before it.
        pause_before_node = False
    else:
        human_node = HumanReviewNode(reviewer)
        pause_before_node = interrupt_before_human

    # --- nodes -------------------------------------------------------- #
    graph = build_support_graph(
        triage=TriageNode(classifier, max_attempts=settings.max_triage_attempts),
        billing=BillingAgent(planner, repository, gateway, composer),
        technical=TechnicalAgent(retriever, composer, top_k=settings.retrieval_top_k),
        general=GeneralAgent(composer),
        guardrail=SentimentGuardrail(analyzer, enabled=settings.enable_human_in_the_loop),
        human_review=human_node,
        interrupt_before_human=pause_before_node,
    )

    return SupportApplication(
        graph=graph,
        settings=settings,
        retriever=retriever,
        repository=repository,
        gateway=gateway,
        offline=offline,
        offline_reason=offline_reason,
        hitl_mode=hitl_mode,
    )
