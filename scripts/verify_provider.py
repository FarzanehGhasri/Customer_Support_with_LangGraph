#!/usr/bin/env python
"""
Check that the configured provider actually answers -- run this before the notebook.

    python scripts/verify_provider.py
    python scripts/verify_provider.py --list-models
    python scripts/verify_provider.py --model gpt-4o-mini

It performs three checks in order, stopping at the first failure so the error
message points at the real problem:

1. configuration is loaded and an API key is present;
2. the endpoint is reachable and the credentials are accepted
   (``GET /models`` -- cheap, and it also tells you which model names are valid);
3. ``with_structured_output`` works against this endpoint, which is the exact
   mechanism the triage agent relies on.

Check 3 matters most: OpenAI-compatible gateways vary in whether they support
function-calling/JSON-schema output, and the triage agent is unusable without it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src`` importable when the script is run directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_system.config import Settings  # noqa: E402
from support_system.domain.schemas import TriageDecision  # noqa: E402
from support_system.infrastructure.classification import LLMIntentClassifier  # noqa: E402
from support_system.infrastructure.llm import LangChainModelProvider  # noqa: E402

OK = "\033[32mOK\033[0m"
FAIL = "\033[31mFAILED\033[0m"


def check_config(settings: Settings) -> bool:
    print("1. Configuration")
    print(f"   {settings.describe()}")
    if not settings.has_credentials:
        print(f"   {FAIL}: no API key. Put it in .env as {settings.api_key_env_var}=...")
        return False
    print(f"   {OK}")
    return True


def check_endpoint(settings: Settings) -> bool:
    """Call ``GET /models`` directly with the OpenAI SDK.

    Done at the SDK level rather than through LangChain because the raw error
    from the gateway (404 vs 401 vs connection refused) is far more diagnostic
    than LangChain's wrapped version.
    """
    print("\n2. Endpoint reachability")
    if settings.provider != "openai":
        print("   skipped (only implemented for OpenAI-compatible endpoints)")
        return True
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url or None,
            timeout=settings.request_timeout,
        )
        models = [m.id for m in client.models.list().data]
    except Exception as exc:  # noqa: BLE001
        print(f"   {FAIL}: {type(exc).__name__}: {exc}")
        print("   -> check SUPPORT_BASE_URL in .env against the provider's quickstart docs.")
        return False

    print(f"   {OK}: endpoint answered with {len(models)} models")
    if settings.model in models:
        print(f"   model '{settings.model}' is available")
    else:
        print(f"   WARNING: '{settings.model}' is NOT in the list. Nearest names:")
        for name in sorted(models)[:15]:
            print(f"     - {name}")
    return True


def list_models(settings: Settings) -> None:
    from openai import OpenAI

    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url or None)
    for name in sorted(m.id for m in client.models.list().data):
        print(name)


def check_structured_output(settings: Settings) -> bool:
    print("\n3. Structured output (with_structured_output) -- required by the triage agent")
    classifier = LLMIntentClassifier(LangChainModelProvider(settings))
    probe = "I was charged twice for my subscription this month."
    decision = classifier.classify(probe)
    if not isinstance(decision, TriageDecision) or decision.confidence == 0.0:
        print(f"   {FAIL}: the model did not return a usable structured decision.")
        print("   -> this gateway may not support JSON-schema / function calling.")
        print("      Try a different model, or fall back to KeywordIntentClassifier.")
        return False
    print(f"   {OK}: '{probe}' -> {decision.department.value} "
          f"(confidence {decision.confidence:.2f})")
    print(f"   reasoning: {decision.reasoning}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="override SUPPORT_MODEL for this run")
    parser.add_argument("--base-url", help="override SUPPORT_BASE_URL for this run")
    parser.add_argument("--list-models", action="store_true", help="print every model id and exit")
    args = parser.parse_args()

    overrides = {}
    if args.model:
        overrides["model"] = args.model
    if args.base_url:
        overrides["base_url"] = args.base_url
    settings = Settings.from_env(**overrides)

    if args.list_models:
        list_models(settings)
        return 0

    for check in (check_config, check_endpoint, check_structured_output):
        if not check(settings):
            print("\nStopped at the first failure above.")
            return 1

    print("\nAll checks passed -- the triage agent is ready to run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
