#!/usr/bin/env python
"""
Run the three assignment scenarios from the terminal.

    python scripts/demo.py                 # all three scenarios
    python scripts/demo.py --scenario 2    # just one
    python scripts/demo.py --offline       # force the deterministic components
    python scripts/demo.py --dynamic       # use graph.interrupt() instead of update_state

For an interactive session where you type your own messages, use
``python scripts/chat.py`` instead.

This is the quickest way to see the system work; the notebook shows the same
thing with explanations.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_system.config import Settings  # noqa: E402
from support_system.graph import build_application  # noqa: E402

ANGRY = "You stole my money! This service is useless! I want to talk to a manager"
MANAGER_REPLY = (
    "من مدیر ارشد هستم. عذرخواهی می‌کنم، مشکل شما را شخصاً پیگیری خواهم کرد. "
    "(I am the senior manager. I apologise; I will personally follow up on your issue.)"
)

BAR = "=" * 78


def show(state, title: str) -> None:
    print(BAR)
    print(title)
    print(BAR)
    print(f"department : {state.get('department')}")
    print(f"sentiment  : {state.get('sentiment')}")
    print(f"escalated  : {state.get('escalated')}")
    print(f"tools used : {state.get('tool_calls') or '(none)'}")
    print("-" * 78)
    for line in state.get("messages", []):
        print("  ", line[:170])
    print("-" * 78)
    print("REPLY TO CUSTOMER:")
    print(state.get("final_response") or "(nothing sent -- waiting for a human)")
    print(BAR + "\n")


def scenario_1(app) -> None:
    state = app.run("How can I reset my password?", user_id="12345", thread_id="s1")
    show(state, "SCENARIO 1 — technical happy path (triage -> technical -> RAG)")


def scenario_2(app) -> None:
    state = app.run(
        "My subscription is not working. My id is 12345", user_id="12345", thread_id="s2"
    )
    show(state, "SCENARIO 2 — billing (triage -> billing -> check_subscription_status)")


def scenario_3(app) -> None:
    state = app.run(ANGRY, user_id="12345", thread_id="s3")
    show(state, "SCENARIO 3 — angry customer: the graph HALTS")
    print(f"interrupted : {app.is_interrupted('s3')}")
    print(f"paused at   : {app.next_nodes('s3')}\n")

    if app.hitl_mode == "dynamic":
        print(">>> resuming with Command(resume=...)  [graph.interrupt() form]\n")
        final = app.resume_with(MANAGER_REPLY, thread_id="s3")
    else:
        print(">>> the manager writes a reply with update_state(), then we resume\n")
        app.inject_manager_reply(MANAGER_REPLY, thread_id="s3")
        final = app.resume("s3")

    show(final, "SCENARIO 3 — resumed after the manager's reply")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3], help="run only this scenario")
    parser.add_argument("--offline", action="store_true",
                        help="force the deterministic components (no API calls, no cost)")
    parser.add_argument("--dynamic", action="store_true",
                        help="use the graph.interrupt() form instead of update_state")
    parser.add_argument("--verbose", action="store_true", help="show each node's decision")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = Settings.from_env()
    app = build_application(
        settings,
        force_offline=args.offline,
        hitl_mode="dynamic" if args.dynamic else "static",
    )

    print()
    if app.offline:
        print(f"MODE: OFFLINE — {app.offline_reason}")
        print("      Answers come from the tools and the knowledge base, not from a model.")
    else:
        print(f"MODE: LIVE — {settings.provider}/{settings.model}")
    print(f"HITL: {app.hitl_mode}\n")

    runners = {1: scenario_1, 2: scenario_2, 3: scenario_3}
    for number in ([args.scenario] if args.scenario else [1, 2, 3]):
        runners[number](app)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
