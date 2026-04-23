"""Run tab: drives the Phase 2 pipeline and renders the resulting brief.

The Arena tab shows per-item evaluator competition; the Run tab is the
brief itself — theme synthesis + the included items summarized from the
winning verdict. HTML download + email continue to work.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from openai import OpenAI

from ..pipeline import run_brief
from ..storage import Storage

__all__ = ["render", "build_report_html"]


def _esc(s: Any) -> str:
    out = str(s or "")
    for a, b in [("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"), ('"', "&quot;")]:
        out = out.replace(a, b)
    return out


def _get_score(it: dict[str, Any]) -> int:
    o = it.get("output") or {}
    return int(o.get("eval_relevance_score") or 0)


def _get_tags(it: dict[str, Any]) -> str:
    tags = (it.get("output") or {}).get("topic_tags") or []
    return ", ".join(tags) if isinstance(tags, list) else str(tags)


def _get_summary(it: dict[str, Any]) -> str:
    o = it.get("output") or {}
    return o.get("clean_summary") or it.get("summary", "")


def _get_why(it: dict[str, Any]) -> str:
    return (it.get("output") or {}).get("why_it_matters", "")


def _get_reason(it: dict[str, Any]) -> str:
    return (it.get("output") or {}).get("action_reason", "")


def build_report_html(
    container: dict[str, Any],
    theme_synthesis: dict[str, Any] | None = None,
) -> str:
    """Build the downloadable / emailable HTML brief."""
    title = container.get("report_title", "SignalScout Research Brief")
    batch_id = container.get("export_batch_id", "")
    generated_at = container.get("generated_at", "")
    items = container.get("included_items", [])

    sorted_items = sorted(items, key=_get_score, reverse=True)
    rows_html = ""
    for idx, it in enumerate(sorted_items):
        rows_html += f"""
        <tr>
          <td style="padding:10px;border-top:1px solid #e5e5e5;vertical-align:top;width:60px;"><b>{_esc(_get_score(it))}</b></td>
          <td style="padding:10px;border-top:1px solid #e5e5e5;vertical-align:top;">
            <div style="font-size:14px;font-weight:700;margin-bottom:4px;">Item {idx + 1}</div>
            <div style="font-size:12px;color:#444;margin-bottom:8px;"><b>Tags:</b> {_esc(_get_tags(it))}</div>
            <div style="margin-bottom:8px;line-height:1.45;"><b>Summary:</b> {_esc(_get_summary(it))}</div>
            <div style="margin-bottom:8px;line-height:1.45;"><b>Why it matters:</b> {_esc(_get_why(it))}</div>
            <div style="color:#555;"><b>Reason:</b> {_esc(_get_reason(it))}</div>
          </td>
        </tr>
        """

    theme_html = ""
    if theme_synthesis:
        themes = theme_synthesis.get("themes") or []
        editorial = theme_synthesis.get("editorial_insight") or ""
        meta = theme_synthesis.get("meta_observation") or ""
        dom = [t for t in themes if isinstance(t, dict)]
        theme_list = "".join(
            f'<li style="margin:6px 0;"><b>{_esc(t.get("name", ""))}</b> — '
            f'{_esc(t.get("evidence", [""])[0] if t.get("evidence") else "")}</li>'
            for t in dom[:8]
        )
        editorial_html = (
            f'<div style="margin-top:12px;"><b>Editorial insight:</b> {_esc(editorial)}</div>'
            if editorial
            else ""
        )
        meta_html = (
            f'<div style="margin-top:10px;color:#555;"><b>Meta observation:</b> {_esc(meta)}</div>'
            if meta
            else ""
        )
        theme_html = f"""
        <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;margin-top:14px;">
          <tr><td style="background:#fff;border:1px solid #e6e6e6;border-radius:12px;padding:14px;">
            <div style="font-size:16px;font-weight:700;margin-bottom:8px;">Weekly Theme Synthesis</div>
            <ul style="margin:6px 0 0 18px;padding:0;">{theme_list}</ul>
            {editorial_html}
            {meta_html}
          </td></tr>
        </table>
        """

    return f"""
    <!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
    <body style="margin:0;font-family:Arial,sans-serif;background:#f6f7f9;color:#111;min-height:100vh;">
    <div style="font-family:Arial,sans-serif;background:#f6f7f9;padding:18px;color:#111;">
      <div style="max-width:900px;margin:0 auto;">
        <div style="background:#fff;border:1px solid #e6e6e6;border-radius:12px;padding:16px;">
          <div style="font-size:20px;font-weight:800;margin-bottom:6px;">{_esc(title)}</div>
          <div style="font-size:12px;color:#666;">
            <div><b>Batch:</b> {_esc(batch_id)}</div>
            <div><b>Generated:</b> {_esc(generated_at)}</div>
            <div><b>Included:</b> {len(items)}</div>
          </div>
        </div>
        {theme_html}
        <div style="height:14px;"></div>
        <div style="background:#fff;border:1px solid #e6e6e6;border-radius:12px;padding:14px;">
          <div style="font-size:16px;font-weight:800;margin-bottom:10px;">Included items</div>
          <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:13px;">
            <thead>
              <tr>
                <th align="left" style="padding:10px;border-bottom:1px solid #e5e5e5;width:60px;">Score</th>
                <th align="left" style="padding:10px;border-bottom:1px solid #e5e5e5;">Details</th>
              </tr>
            </thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>
      </div>
    </div>
    </body></html>
    """


def _render_native(container: dict[str, Any]) -> None:
    st.subheader(container.get("report_title", "SignalScout Research Brief"))
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Batch", container.get("export_batch_id", "—"))
    with col2:
        generated = container.get("generated_at", "—")
        st.metric("Generated", generated[:19].replace("T", " "))
    with col3:
        st.metric("Included", container.get("included_count", 0))

    theme = container.get("theme_synthesis") or {}
    if theme:
        with st.expander("Theme synthesis", expanded=True):
            for t in (theme.get("themes") or [])[:8]:
                if isinstance(t, dict):
                    evidence = t.get("evidence") or [""]
                    first = evidence[0] if evidence else ""
                    st.markdown(f"**{t.get('name', '')}** — {first}")
            if theme.get("editorial_insight"):
                st.markdown(f"**Editorial insight:** {theme['editorial_insight']}")
            if theme.get("meta_observation"):
                st.markdown(f"*Meta observation:* {theme['meta_observation']}")

    items = container.get("included_items") or []
    st.markdown("---")
    st.subheader("Included items")
    st.caption("Score shown is the winning evaluator's. Open the Arena tab to see all three verdicts side-by-side.")
    if not items:
        st.info("No items scored ≥ 70 this run. Try again or widen the limits.")
        return

    sorted_items = sorted(items, key=_get_score, reverse=True)
    for it in sorted_items:
        score = _get_score(it)
        title = (it.get("title") or "Item")[:80]
        with st.expander(f"Score **{score}** — {title}"):
            st.markdown(f"**Tags:** {_get_tags(it)}")
            st.markdown(f"**Summary:** {_get_summary(it)}")
            st.markdown(f"**Why it matters:** {_get_why(it)}")
            reason = _get_reason(it)
            if reason:
                st.markdown(f"*{reason}*")
            if it.get("url"):
                st.markdown(f"[Open source ↗]({it['url']})")


def render(
    *,
    client: OpenAI | None,
    storage: Storage,
    user_id: str,
    arxiv_limit: int,
    smol_limit: int,
    max_evaluate: int,
) -> None:
    """Render the Run tab and wire the Run button to ``pipeline.run_brief``."""
    st.markdown(
        "Three evaluators — Skeptic, Scout, Operator — score each item in parallel. "
        "The Run tab shows the brief; open **Arena** to watch the competition per item."
    )

    if not user_id:
        st.warning("Enter your email in the sidebar to enable learning across runs.")
        return

    if st.button("Run workflow", type="primary"):
        if client is None:
            st.error(
                "No OpenAI API key. Use the default key (from app config) "
                'or choose "Use my own API key" in the sidebar.'
            )
            return

        progress = st.progress(0, text="Starting pipeline…")

        def _on_progress(done: int, total: int, label: str) -> None:
            progress.progress(done / max(1, total), text=label)

        with st.spinner("Running pipeline…"):
            container = run_brief(
                client=client,
                storage=storage,
                user_id=user_id,
                arxiv_limit=arxiv_limit,
                smol_limit=smol_limit,
                max_evaluate=max_evaluate,
                on_progress=_on_progress,
            )
        progress.empty()

        if container.get("error"):
            st.warning(container["error"])
            return

        html = build_report_html(container, container.get("theme_synthesis") or {})
        st.session_state.last_report_html = html
        st.session_state.last_report_subject = (
            f"{container.get('report_title', 'Research Brief')} — "
            f"{container.get('export_batch_id', '')}"
        )
        st.session_state.last_container = container

        st.success(
            f"Done. Evaluated **{container.get('evaluated_count', 0)}** items, "
            f"**{container.get('included_count', 0)}** included."
        )

    container = st.session_state.get("last_container")
    if container:
        _render_native(container)
        with st.expander("HTML preview / export"):
            html = st.session_state.get("last_report_html", "")
            st.components.v1.html(html, height=700, scrolling=True)
            st.download_button(
                "Download report (HTML)",
                html,
                file_name=f"signalscout_brief_{container.get('export_batch_id', 'report')}.html",
                mime="text/html",
            )
