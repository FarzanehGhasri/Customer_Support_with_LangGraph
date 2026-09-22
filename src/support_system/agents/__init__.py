"""Graph nodes ("agents"). Each one implements the SupportNode Protocol."""

from .base import BaseSupportNode
from .triage import TriageNode

__all__ = ["BaseSupportNode", "TriageNode"]
