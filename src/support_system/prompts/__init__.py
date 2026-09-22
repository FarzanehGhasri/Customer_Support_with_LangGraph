"""Prompt templates, kept out of the agent classes so they can be tuned freely."""

from .triage import TRIAGE_SYSTEM_PROMPT, TRIAGE_USER_PROMPT, build_history_block

__all__ = ["TRIAGE_SYSTEM_PROMPT", "TRIAGE_USER_PROMPT", "build_history_block"]
