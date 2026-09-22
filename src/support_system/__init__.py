"""
Multi-agent customer-support system built with LangGraph.

Layering (outermost depends on innermost, never the reverse)::

    notebooks / graph  ->  agents  ->  interfaces  ->  domain
                             ^
                             |
                       infrastructure  (implements interfaces)

``domain`` and ``interfaces`` are dependency-free (Pydantic only), so importing
this package never requires LangChain or an API key.  Heavy modules are
imported on demand by the layer that needs them.
"""

__version__ = "0.1.0"  # Step 1: foundations

__all__ = ["__version__"]
