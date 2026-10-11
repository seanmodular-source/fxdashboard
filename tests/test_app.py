"""Smoke test: the whole app renders every tab and analytics view without an exception (offline data)."""
import os

import pytest
from streamlit.testing.v1 import AppTest

from conftest import ROOT


@pytest.fixture(scope="module")
def app():
    os.environ["FXDASH_OFFLINE"] = "1"
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    return at


def test_app_renders_without_errors(app):
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    assert len(app.tabs) == 3


@pytest.mark.parametrize("view", ["Rich / cheap", "Seasonality", "DXY beta & peers", "Sessions"])
def test_analytics_views_render(app, view):
    app.session_state["ana_view"] = view
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]


def test_trades_survive_reruns(app):
    n = len(app.session_state["trades"])
    app.run()
    app.run()
    assert len(app.session_state["trades"]) == n
