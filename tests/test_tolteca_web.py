"""Tests for `tolteca_web` package."""

import tolteca_web


def test_import():
    """Verify the package can be imported."""
    assert tolteca_web


def test_sweep_viewer_imports_with_supported_plotly_api():
    """The sweep viewer must not depend on private Plotly helpers."""
    from tolteca_web.sweep import SweepViewerPage

    assert SweepViewerPage
