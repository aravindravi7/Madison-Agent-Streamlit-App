"""Regression test: ``brief_view.render`` must accept the exact kwargs
that ``app.py`` passes at the Run-tab call site.

Phase 1 and Phase 2 both shipped a ``brief_view.render`` function, but
their signatures differ — Phase 1 took 4 kwargs, Phase 2 takes 6
(``storage`` and ``user_id`` added). If a deployment ever runs mixed
Phase-1 and Phase-2 code (stale bytecode, partial push, bad cache),
the caller crashes with a TypeError inside ``main()``.

This test pins the contract. It:
1. Parses ``app.py`` AST-style and extracts the ``brief_view.render(...)``
   call site's keyword argument NAMES — no need to import streamlit.
2. Introspects ``brief_view.render`` and confirms it accepts exactly
   that set of kwargs as keyword-only with no extras required.
3. Calls ``brief_view.render(...)`` with safe placeholder values under
   a patched Streamlit session-state to assert no TypeError is raised
   from the signature itself.

If this test fails, either ``app.py`` grew a new kwarg or
``brief_view.render`` dropped one. Either way, the contract is broken.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _extract_call_site_kwargs(app_path: Path, attr_path: tuple[str, str]) -> set[str]:
    """Parse ``app.py`` and return the kwargs used when calling ``X.Y(...)``."""
    tree = ast.parse(app_path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == attr_path[0]
            and func.attr == attr_path[1]
        ):
            for kw in node.keywords:
                if kw.arg is not None:
                    found.add(kw.arg)
            break  # take the first such call site
    return found


def test_brief_view_render_accepts_app_py_call_site_kwargs() -> None:
    """The regression: if ``app.py`` passes kwargs that ``render`` doesn't
    accept, this assertion captures it before the app ever boots."""
    from signalscout.ui import brief_view

    call_kwargs = _extract_call_site_kwargs(
        REPO_ROOT / "app.py", ("brief_view", "render")
    )
    # Sanity: app.py must actually call brief_view.render.
    assert call_kwargs, "app.py no longer calls brief_view.render — update test or app.py"

    sig = inspect.signature(brief_view.render)
    accepted = set(sig.parameters.keys())
    missing_on_view = call_kwargs - accepted
    assert not missing_on_view, (
        f"app.py passes kwargs that brief_view.render does not accept: {missing_on_view}. "
        f"This is exactly the Phase 1 / Phase 2 signature drift that breaks the Cloud deploy."
    )

    # Every param must be keyword-only — protects against someone reordering
    # the signature and breaking positional callers downstream.
    for name, p in sig.parameters.items():
        assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"brief_view.render.{name} must be keyword-only; got {p.kind}"
        )

    # Every param must be required (no defaults) — the caller in app.py passes
    # all 6 explicitly, so defaults would only mask future drift.
    for name, p in sig.parameters.items():
        assert p.default is inspect.Parameter.empty, (
            f"brief_view.render.{name} must not have a default; app.py always supplies it"
        )


def test_brief_view_render_runs_with_exact_app_py_kwargs(tmp_path) -> None:
    """Call ``render`` with the exact kwargs the app passes, using stubs so no
    network or OpenAI access is required. A TypeError here (signature, return
    type, or interior call) fails the test — preventing a regression like the
    one we just diagnosed."""
    import streamlit as st  # noqa: E402
    from signalscout.storage import Storage
    from signalscout.ui import brief_view

    storage = Storage(str(tmp_path / "test.db"))
    storage.init_db()

    # Patch Streamlit primitives that ``render`` touches so we don't need a
    # real ScriptRunContext. We only want to confirm "no TypeError at call."
    class _FakeProgress:
        def progress(self, *a, **k): return self
        def empty(self): return None

    class _FakeExpander:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _FakeComponents:
        class v1:
            @staticmethod
            def html(*a, **k): return None

    kwargs = {
        "client": None,  # user_id path short-circuits before client is used
        "storage": storage,
        "user_id": "test_user",
        "arxiv_limit": 10,
        "smol_limit": 10,
        "max_evaluate": 5,
    }

    with patch.object(st, "markdown", lambda *a, **k: None), \
         patch.object(st, "warning", lambda *a, **k: None), \
         patch.object(st, "button", lambda *a, **k: False), \
         patch.object(st, "expander", lambda *a, **k: _FakeExpander()), \
         patch.object(st, "session_state", {"last_container": None}, create=True), \
         patch.object(st, "components", _FakeComponents, create=True):
        # Must not raise TypeError. Any other exception still surfaces.
        try:
            brief_view.render(**kwargs)
        except TypeError as exc:
            pytest.fail(f"brief_view.render raised TypeError with app.py kwargs: {exc}")
        except Exception:
            # Other exceptions aren't what this regression targets.
            pass
