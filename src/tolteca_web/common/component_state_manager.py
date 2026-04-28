"""Backward-compatible alias — re-exports :class:`ComponentStateManager`.

New code should import directly from ``state_manager`` or
``url_state_manager`` as appropriate.
"""

from __future__ import annotations

from .state_manager import ComponentStateManager

__all__ = ["ComponentStateManager"]
