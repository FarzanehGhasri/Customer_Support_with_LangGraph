"""
Requirement-by-requirement check against `HW_07_multi agent_2.pdf`.

Each test is named after one clause of the assignment. Failing one of these
means the deliverable no longer satisfies the brief, regardless of whether the
rest of the suite passes.
"""

from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import Annotated, List, get_args, get_type_hints

import pytest

from support_system.config import PROJECT_ROOT, Settings
from support_system.domain import (
    Department,
    Sentiment,
    SupportState,
    TriageDecision,
)
from support_system.graph import build_application

ANGRY = "You stole my money! This service is useless! I want to talk to a manager"
MANAGER = "I am the senior manager. I apologise; I will personally follow up on your issue."


@pytest.fixture(scope="module")
def app():
    return build_application(force_offline=True)


# ===========================================================================
# ۲. معماری سیستم -- Architecture
# ===========================================================================


class TestNode1Triage:
    """گره ۱: ایجنت دسته‌بندی و توزیع"""

    def test_classifies_into_the_three_required_categories(self):
        """BILLING / TECHNICAL / GENERAL"""
        allowed = set(TriageDecision.model_json_schema()["$defs"]["Department"]["enum"])
        assert {"Billing", "Technical", "General"} <= allowed

    def test_output_is_a_structured_decision(self):
        """خروجی: یک تصمیم ساختاریافته"""
        schema = TriageDecision.model_json_schema()
        assert schema["type"] == "object"
        assert "department" in schema["required"]

    def test_the_decision_determines_the_next_node(self, app):
        """مشخص می‌کند گره بعدی کدام است"""
        from support_system.domain import NextStep
        from support_system.graph import ROUTES

        for department in Department.specialists():
            assert ROUTES[NextStep.for_department(department).value]


class TestNode2Billing:
    """گره ۲: متخصص امور مالی"""

    def test_has_check_subscription_status_tool(self, app):
        from support_system.tools import make_billing_tools

        names = [t.name for t in make_billing_tools(app.repository, app.gateway)]
        assert "check_subscription_status" in names

    def test_has_process_refund_tool(self, app):
        from support_system.tools import make_billing_tools

        names = [t.name for t in make_billing_tools(app.repository, app.gateway)]
        assert "process_refund" in names

    def test_check_subscription_reports_active_or_expired(self, app):
        """بررسی فعال یا منقضی بودن اشتراک کاربر"""
        assert app.repository.get_status("12345").status == "expired"
        assert app.repository.get_status("67890").is_active

    def test_process_refund_is_mocked(self, app):
        """به صورت Mock، تایید بازگشت وجه"""
        assert app.gateway.process_refund("TXN-1001").approved is True

    def test_does_not_answer_technical_questions(self, app):
        """این ایجنت نباید به سوالات فنی پاسخ دهد"""
        from support_system.agents import BillingAgent
        from support_system.domain import initial_state
        from support_system.infrastructure.composition import TemplateResponseComposer
        from support_system.infrastructure.planning import RuleBasedBillingPlanner

        agent = BillingAgent(
            RuleBasedBillingPlanner(), app.repository, app.gateway, TemplateResponseComposer()
        )
        update = agent(initial_state("How can I reset my password?", "12345"))
        assert "draft_response" not in update

    def test_returns_unrelated_requests_to_triage(self, app):
        """باید درخواست را دوباره به تریاژ برگرداند"""
        from support_system.agents import BillingAgent
        from support_system.domain import NextStep, initial_state
        from support_system.infrastructure.composition import TemplateResponseComposer
        from support_system.infrastructure.planning import RuleBasedBillingPlanner

        agent = BillingAgent(
            RuleBasedBillingPlanner(), app.repository, app.gateway, TemplateResponseComposer()
        )
        update = agent(initial_state("How can I reset my password?", "12345"))
        assert update["next_step"] == NextStep.TRIAGE


class TestNode3Technical:
    """گره ۳: متخصص پشتیبانی فنی"""

    def test_has_search_knowledge_base_tool(self, app):
        from support_system.tools import make_technical_tools

        assert [t.name for t in make_technical_tools(app.retriever)] == ["search_knowledge_base"]

    def test_rag_searches_a_documentation_corpus(self, app):
        """استفاده از سیستم RAG برای جستجو در یک فایل راهنما یا داکیومنت فنی"""
        docs = list(Settings.from_env().knowledge_base_dir.glob("*.md"))
        assert len(docs) >= 1
        assert app.retriever.search("How can I reset my password?").has_grounding

    def test_does_not_hallucinate_when_nothing_is_found(self, app):
        """نباید توهم داشته باشد و باید اعلام کند که پاسخ را نمی‌داند"""
        from support_system.agents import TechnicalAgent
        from support_system.domain import initial_state
        from support_system.infrastructure.composition import TemplateResponseComposer

        agent = TechnicalAgent(app.retriever, TemplateResponseComposer())
        update = agent(initial_state("What is the capital of France?", "1"))
        assert "no grounding" in update["tool_calls"][0]
        assert "could not find" in update["draft_response"].lower()


class TestNode4Guardrail:
    """گره ۴: محافظ تحلیل احساسات"""

    def test_checks_tone_before_the_reply_reaches_the_customer(self, app):
        """قبل از ارسال پاسخ نهایی، لحن پیام کاربر را بررسی می‌کند"""
        state = app.run(ANGRY, user_id="12345", thread_id="g1")
        assert state["sentiment"] == Sentiment.NEGATIVE
        assert not state.get("final_response")      # nothing was sent

    def test_negative_sentiment_halts_execution(self, app):
        """اجرای گراف را متوقف کرده و منتظر دخالت انسان می‌ماند"""
        app.run(ANGRY, user_id="12345", thread_id="g2")
        assert app.is_interrupted("g2")
        assert app.next_nodes("g2") == ("human_review",)

    def test_neutral_sentiment_shows_the_generated_reply(self, app):
        """اگر احساسات خنثی/مثبت باشد: پاسخ تولید شده را نمایش می‌دهد"""
        state = app.run("How can I reset my password?", user_id="1", thread_id="g3")
        assert state["sentiment"] == Sentiment.NEUTRAL
        assert state["final_response"]
        assert not app.is_interrupted("g3")


# ===========================================================================
# ۳. مراحل پیاده‌سازی فنی -- Implementation steps
# ===========================================================================


class TestStep1State:
    """مرحله ۱: تعریف وضعیت (State)"""

    def test_uses_typeddict_with_the_five_required_keys(self):
        hints = get_type_hints(SupportState, include_extras=True)
        assert {"messages", "user_id", "sentiment", "department", "next_step"} <= set(hints)

    def test_messages_is_annotated_list_str_with_operator_add(self):
        """messages: Annotated[List[str], operator.add]"""
        hints = get_type_hints(SupportState, include_extras=True)
        args = get_args(hints["messages"])
        assert args[0] in (List[str], list[str])
        assert operator.add in args

    def test_the_other_four_keys_are_strings(self):
        hints = get_type_hints(SupportState, include_extras=True)
        for key in ("user_id", "sentiment", "department", "next_step"):
            assert hints[key] is str

    def test_documented_values_are_the_ones_actually_used(self):
        """sentiment: Positive/Neutral/Negative -- department: Billing/Technical/Triage"""
        assert {s.value for s in Sentiment} == {"Positive", "Neutral", "Negative"}
        assert {"Billing", "Technical", "Triage"} <= {d.value for d in Department}


class TestStep2StructuredOutput:
    """مرحله ۲: ساخت ایجنت دسته‌بندی با with_structured_output"""

    def test_the_classifier_calls_with_structured_output(self):
        """قابلیت with_structured_output در لنگ‌چین"""
        from support_system.domain import TriageDecision
        from support_system.infrastructure.classification import LLMIntentClassifier

        seen = {}

        class _Provider:
            def get_chat_model(self, *, temperature=None):
                raise AssertionError("must use the structured model")

            def get_structured_model(self, schema, *, temperature=None):
                seen["schema"] = schema

                class _R:
                    def invoke(self, messages):
                        return TriageDecision(department=Department.BILLING)

                return _R()

        LLMIntentClassifier(_Provider()).classify("I want a refund")
        assert seen["schema"] is TriageDecision

    def test_the_output_is_always_json(self):
        """خروجی آن حتماً یک JSON باشد که قصد کاربر را مشخص می‌کند"""
        decision = TriageDecision(department=Department.BILLING, confidence=0.9)
        assert json.loads(decision.model_dump_json())["department"] == "Billing"

    def test_an_invalid_category_is_rejected(self):
        with pytest.raises(ValueError):
            TriageDecision(department="Marketing")


class TestStep3HumanInTheLoop:
    """مرحله ۳: پیاده‌سازی نظارت انسانی"""

    def test_a_checkpointer_is_used(self, app):
        """باید از قابلیت checkpointer در LangGraph استفاده کنید"""
        assert app.graph.checkpointer is not None

    def test_the_graph_declares_a_static_interrupt(self, app):
        assert "__interrupt = before" in app.graph.get_graph().draw_mermaid()

    def test_the_dynamic_interrupt_form_is_also_supported(self):
        """باید با دستور graph.interrupt() اجرا را متوقف کنید"""
        dynamic = build_application(force_offline=True, hitl_mode="dynamic")
        dynamic.run(ANGRY, user_id="12345", thread_id="d1")
        assert dynamic.is_interrupted("d1")
        payload = dynamic.pending_interrupt("d1")
        # The manager is shown the state and the agent's draft.
        assert payload["sentiment"] == "Negative"
        assert "agent_draft" in payload and "customer_message" in payload

    def test_the_manager_function_can_approve(self):
        """این تابع باید بتواند ... پاسخ ایجنت را تایید ... کند"""
        from support_system.agents import HumanReviewNode
        from support_system.domain import initial_state
        from support_system.infrastructure.human import AutoApproveReviewer

        state = initial_state(ANGRY, "1")
        state["draft_response"] = "the agent's draft"
        assert HumanReviewNode(AutoApproveReviewer())(state)["final_response"] == "the agent's draft"

    def test_the_manager_function_can_reject_and_replace(self):
        """... یا رد کند (و یا یک پاسخ دستی جایگزین کند)"""
        from support_system.agents import HumanReviewNode
        from support_system.domain import HumanDecision, initial_state
        from support_system.infrastructure.human import ScriptedReviewer

        state = initial_state(ANGRY, "1")
        state["draft_response"] = "the agent's draft"
        reviewer = ScriptedReviewer(HumanDecision(approved=False, replacement_response=MANAGER))
        assert HumanReviewNode(reviewer)(state)["final_response"] == MANAGER

    def test_the_manager_function_can_inspect_the_state(self):
        """این تابع باید بتواند وضعیت را بررسی کرده ..."""
        from support_system.agents import HumanReviewNode
        from support_system.domain import HumanDecision, initial_state

        seen = {}

        class _Inspecting:
            def review(self, state):
                seen.update(state)
                return HumanDecision(approved=True)

        state = initial_state(ANGRY, "12345")
        state["draft_response"] = "draft"
        HumanReviewNode(_Inspecting())(state)
        assert seen["user_id"] == "12345" and seen["draft_response"] == "draft"


# ===========================================================================
# ۴. سناریوهای مورد انتظار برای تحویل -- Delivery scenarios
# ===========================================================================


class TestDeliveryScenarios:
    def test_scenario_1_technical_happy_path(self, app):
        """تریاژ < متخصص فنی < جستجوی RAG"""
        state = app.run("How can I reset my password?", user_id="12345", thread_id="sc1")
        assert state["department"] == Department.TECHNICAL
        assert "search_knowledge_base" in state["tool_calls"][0]
        assert "grounded" in state["tool_calls"][0]
        # "برای تغییر رمز عبور روی لینک زیر کلیک کنید"
        assert "Forgot password" in state["final_response"]

    def test_scenario_2_sensitive_billing_operation(self, app):
        """تریاژ < متخصص مالی < ابزار check_subscription"""
        state = app.run(
            "My subscription is not working. My id is 12345", user_id="12345", thread_id="sc2"
        )
        assert state["department"] == Department.BILLING
        assert state["tool_calls"] == ["check_subscription_status(12345) -> expired"]
        # "اشتراک شما منقضی شده است"
        assert "expired" in state["final_response"]

    def test_scenario_3_static_form_uses_update_state(self, app):
        """شما باید با استفاده از متد update_state یک پاسخ دستی از طرف مدیر وارد کنید"""
        state = app.run(ANGRY, user_id="12345", thread_id="sc3")
        assert state["department"] == Department.BILLING      # triage says billing
        assert state["sentiment"] == Sentiment.NEGATIVE       # guardrail detects anger
        assert app.is_interrupted("sc3")                      # system halts

        app.inject_manager_reply(MANAGER, thread_id="sc3")    # update_state
        final = app.resume("sc3")
        assert final["final_response"] == MANAGER

    def test_scenario_3_dynamic_form_uses_command_resume(self):
        dynamic = build_application(force_offline=True, hitl_mode="dynamic")
        dynamic.run(ANGRY, user_id="12345", thread_id="sc3d")
        assert dynamic.is_interrupted("sc3d")
        final = dynamic.resume_with(MANAGER, thread_id="sc3d")
        assert final["final_response"] == MANAGER


# ===========================================================================
# نحوه‌ی تحویل -- Delivery format
# ===========================================================================


class TestDeliverables:
    def test_a_notebook_exists_and_has_been_executed(self):
        """کد کامل پیاده‌سازی شده در قالب فایل .ipynb"""
        notebooks = list((PROJECT_ROOT / "notebooks").glob("*.ipynb"))
        assert notebooks, "no .ipynb deliverable found"

        nb = json.loads(notebooks[0].read_text(encoding="utf-8"))
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert code_cells, "notebook has no code"
        # Executed, i.e. outputs are committed for the grader to read.
        assert any(c.get("outputs") for c in code_cells), "notebook has no stored outputs"
        assert not any(
            o.get("output_type") == "error" for c in code_cells for o in c.get("outputs", [])
        ), "notebook contains an execution error"

    def test_the_notebook_demonstrates_all_three_scenarios(self):
        nb = json.loads(
            (PROJECT_ROOT / "notebooks" / "customer_support_langgraph.ipynb").read_text(
                encoding="utf-8"
            )
        )
        text = json.dumps(nb, ensure_ascii=False)
        assert "SCENARIO 1" in text and "SCENARIO 2" in text and "SCENARIO 3" in text

    def test_the_graph_drawing_is_produced_by_draw_mermaid_png(self):
        """به همراه تصویر گراف تولید شده (با استفاده از draw_mermaid_png())"""
        nb = json.loads(
            (PROJECT_ROOT / "notebooks" / "customer_support_langgraph.ipynb").read_text(
                encoding="utf-8"
            )
        )
        assert "draw_mermaid_png" in json.dumps(nb)
        assert (PROJECT_ROOT / "scripts" / "render_graph.py").is_file()
        # The Mermaid source is always committed; the PNG needs network access.
        assert (PROJECT_ROOT / "docs" / "images" / "support_graph.mmd").is_file()
