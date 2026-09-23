"""
The three tools named in the assignment, as real LangChain tool objects.

    check_subscription_status(user_id)
    process_refund(transaction_id)
    search_knowledge_base(query)

Each is built by a *factory* that takes the implementation as an argument and
closes over it. That matters: a bare ``@tool`` function would have to reach for
a global repository, which is exactly the hidden coupling the rest of this
project avoids. With a factory the tool keeps its clean LLM-facing signature
while the data source stays injected.

The docstring of each tool is sent to the model as the tool description, so it
is written for the model, not for us.
"""

from __future__ import annotations

from typing import List

from langchain_core.tools import StructuredTool

from ..interfaces.billing import RefundGateway, SubscriptionRepository
from ..interfaces.retrieval import KnowledgeRetriever


def make_check_subscription_tool(repository: SubscriptionRepository) -> StructuredTool:
    """Build the ``check_subscription_status`` tool over ``repository``."""

    def check_subscription_status(user_id: str) -> str:
        """Look up a customer's subscription. Use this whenever the customer asks
        about their plan, renewal, or whether their subscription is active.
        Pass the customer's numeric account id, e.g. '12345'."""
        return repository.get_status(user_id).summary()

    return StructuredTool.from_function(
        func=check_subscription_status,
        name="check_subscription_status",
        description=check_subscription_status.__doc__,
    )


def make_process_refund_tool(gateway: RefundGateway) -> StructuredTool:
    """Build the ``process_refund`` tool over ``gateway``."""

    def process_refund(transaction_id: str) -> str:
        """Refund a specific transaction. Only call this when the customer has
        explicitly asked for a refund AND has given a transaction id such as
        'TXN-1001'. Never guess a transaction id."""
        return gateway.process_refund(transaction_id).summary()

    return StructuredTool.from_function(
        func=process_refund,
        name="process_refund",
        description=process_refund.__doc__,
    )


def make_search_knowledge_base_tool(
    retriever: KnowledgeRetriever, *, top_k: int = 3
) -> StructuredTool:
    """Build the ``search_knowledge_base`` (RAG) tool over ``retriever``."""

    def search_knowledge_base(query: str) -> str:
        """Search the product documentation for an answer to a technical
        question. Returns the relevant passages, or a statement that nothing was
        found. If nothing is found you must tell the customer you do not know --
        never invent an answer."""
        result = retriever.search(query, top_k=top_k)
        if not result.has_grounding:
            return (
                "NO_RESULTS: the documentation contains nothing relevant to this "
                "question. Tell the customer you do not know and offer to escalate."
            )
        return result.as_context()

    return StructuredTool.from_function(
        func=search_knowledge_base,
        name="search_knowledge_base",
        description=search_knowledge_base.__doc__,
    )


def make_billing_tools(
    repository: SubscriptionRepository, gateway: RefundGateway
) -> List[StructuredTool]:
    """Every tool the Billing specialist is allowed to use."""
    return [
        make_check_subscription_tool(repository),
        make_process_refund_tool(gateway),
    ]


def make_technical_tools(retriever: KnowledgeRetriever) -> List[StructuredTool]:
    """Every tool the Technical specialist is allowed to use.

    Note what is *absent*: no billing tool is in this list. Tool access is how
    the spec's "this agent must not answer questions outside its area" rule is
    enforced structurally, not just by asking the prompt nicely.
    """
    return [make_search_knowledge_base_tool(retriever)]
