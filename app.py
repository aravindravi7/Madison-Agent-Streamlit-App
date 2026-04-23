"""SignalScout — Streamlit entry point.

Thin shell. Routes sidebar config + three tabs to view modules. Pipeline,
storage, sources, evaluators all live under ``signalscout/``.
"""

from __future__ import annotations

import hashlib

import streamlit as st
from openai import OpenAI
from streamlit.errors import StreamlitSecretNotFoundError

from signalscout.email_brief import send_brief_email
from signalscout.storage import Storage
from signalscout.ui import arena_view, brief_view, learning_view
from signalscout.ui.theme import inject_theme


def _secret(*keys: str) -> str | None:
    try:
        for k in keys:
            v = st.secrets.get(k)
            if v:
                return str(v)
    except StreamlitSecretNotFoundError:
        pass
    return None


def _gmail_creds() -> tuple[str, str] | None:
    user = _secret("GMAIL_USER", "gmail_user")
    pwd = _secret("GMAIL_APP_PASSWORD", "gmail_app_password")
    return (user, pwd) if user and pwd else None


def _get_client(key_source: str, own_key: str) -> OpenAI | None:
    api_key = own_key if key_source == "own" else _secret("OPENAI_API_KEY")
    return OpenAI(api_key=api_key) if api_key else None


def _hash_email(email: str) -> str:
    return hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()[:16]


def _ensure_session_state() -> None:
    st.session_state.setdefault("user_email", "")
    st.session_state.setdefault("user_id", "")
    st.session_state.setdefault("openai_key_source", "default")
    st.session_state.setdefault("openai_api_key", "")
    st.session_state.setdefault("last_report_html", None)
    st.session_state.setdefault("last_report_subject", "")
    st.session_state.setdefault("last_report_container", None)


def _render_sidebar() -> tuple[OpenAI | None, int, int, int]:
    st.sidebar.header("Settings")

    email = st.sidebar.text_input(
        "Your email (establishes a stable user id for learning)",
        value=st.session_state.user_email,
        placeholder="you@example.com",
        help="Hashed locally. The bandit persists its state against this id.",
    )
    if email and email != st.session_state.user_email:
        st.session_state.user_email = email
        st.session_state.user_id = _hash_email(email)
    if st.session_state.user_id:
        st.sidebar.caption(f"user_id: `{st.session_state.user_id}`")

    st.sidebar.markdown("---")
    key_source = st.sidebar.radio(
        "OpenAI API Key",
        options=["default", "own"],
        format_func=lambda x: "Use default key (app config)" if x == "default" else "Use my own key",
        index=0 if st.session_state.openai_key_source == "default" else 1,
    )
    st.session_state.openai_key_source = key_source
    own_key = ""
    if key_source == "own":
        own_key = st.sidebar.text_input("Your OpenAI API Key", type="password")
        st.session_state.openai_api_key = own_key
    else:
        st.session_state.openai_api_key = ""

    client = _get_client(key_source, own_key)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Data source limits")
    arxiv_limit = st.sidebar.number_input("arXiv cs.AI limit", 1, 150, 50, 5)
    smol_limit = st.sidebar.number_input("Smol RSS limit", 1, 150, 50, 5)
    max_evaluate = st.sidebar.number_input("Max items to evaluate", 1, 320, 25, 1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Send brief by email")
    recipient = st.sidebar.text_input("Recipient", placeholder="you@example.com")
    creds = _gmail_creds()
    has_report = bool(st.session_state.last_report_html)
    if not creds:
        st.sidebar.caption("Add GMAIL_USER + GMAIL_APP_PASSWORD to secrets.toml to enable.")
    elif not has_report:
        st.sidebar.caption("Generate a brief first to enable sending.")
    if st.sidebar.button("Send brief to my email", disabled=not creds or not has_report):
        if creds and recipient:
            ok, msg = send_brief_email(
                recipient,
                st.session_state.last_report_html or "",
                st.session_state.last_report_subject,
                creds,
            )
            (st.sidebar.success if ok else st.sidebar.error)(msg)

    return client, int(arxiv_limit), int(smol_limit), int(max_evaluate)


def main() -> None:
    st.set_page_config(page_title="SignalScout", page_icon="📡", layout="wide")
    inject_theme()
    _ensure_session_state()

    storage = Storage()
    storage.init_db()

    st.title("📡 SignalScout")
    st.caption("Your taste, on autopilot.")

    client, arxiv_limit, smol_limit, max_evaluate = _render_sidebar()

    run_tab, arena_tab, learning_tab = st.tabs(["Run", "Arena", "Learning"])
    with run_tab:
        brief_view.render(
            client=client,
            arxiv_limit=arxiv_limit,
            smol_limit=smol_limit,
            max_evaluate=max_evaluate,
        )
    with arena_tab:
        arena_view.render()
    with learning_tab:
        learning_view.render()


main()
