"""SignalScout — Streamlit entry point.

Thin shell. Routes sidebar config + three tabs to view modules. Pipeline,
storage, sources, evaluators all live under ``signalscout/``.
"""

from __future__ import annotations

import hashlib

import streamlit as st
from openai import OpenAI
from streamlit.errors import StreamlitSecretNotFoundError

from signalscout.bandit import TasteBandit
from signalscout.demo_seed import DEMO_USER_ID, seed_demo_user
from signalscout.email_brief import send_brief_email
from signalscout.evaluators import EVALUATORS
from signalscout.storage import Storage
from signalscout.ui import arena_view, brief_view, learning_view
from signalscout.ui.theme import inject_theme
_DEFAULTS = {
    "user_email": "", "user_id": "", "openai_key_source": "default",
    "demo_mode": False, "last_report_html": None,
    "last_report_subject": "", "last_container": None,
}


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
    if key_source == "own" and own_key:
        return OpenAI(api_key=own_key)
    default = _secret("OPENAI_API_KEY")
    return OpenAI(api_key=default) if default else None


def _hash_email(email: str) -> str:
    return hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()[:16]


def _render_sidebar(storage: Storage) -> tuple[OpenAI | None, str, int, int, int]:
    st.sidebar.markdown(
        "<div style=\"font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:24px;"
        "line-height:1.1;margin-bottom:10px;color:#EAEAEA;\">"
        "<span style=\"color:#00F5D4;margin-right:6px;\">📡</span>SignalScout</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.header("Settings")
    demo_mode = st.sidebar.toggle(
        "Demo mode", value=st.session_state.demo_mode,
        help="Locks user_id to `demo_user` and uses the default key.",
    )
    st.session_state.demo_mode = demo_mode
    if demo_mode:
        user_id = DEMO_USER_ID
        st.sidebar.caption(f"user_id: `{user_id}`")
        if st.sidebar.button("Seed demo data", help="Populate demo_user with 30 decisions + feedback."):
            n = seed_demo_user(storage)
            if n > 0:
                st.sidebar.success(f"Seeded {n} decisions for demo_user. Open Arena + Learning tabs to see the state.")
            else:
                st.sidebar.info("Demo data already present.")
    else:
        email = st.sidebar.text_input(
            "Your email (establishes a stable user id for learning)",
            value=st.session_state.user_email, placeholder="you@example.com",
        )
        if email and email != st.session_state.user_email:
            st.session_state.user_email = email
            st.session_state.user_id = _hash_email(email)
        user_id = st.session_state.user_id
        if user_id:
            st.sidebar.caption(f"user_id: `{user_id}`")

    st.sidebar.markdown("---")
    key_source = st.sidebar.radio(
        "OpenAI API Key", options=["default", "own"],
        format_func=lambda x: "Use default key (app config)" if x == "default" else "Use my own key",
        index=0 if st.session_state.openai_key_source == "default" else 1,
        disabled=demo_mode, help="Demo mode forces the default key.",
    )
    st.session_state.openai_key_source = key_source if not demo_mode else "default"
    own_key = ""
    if key_source == "own" and not demo_mode:
        own_key = st.sidebar.text_input("Your OpenAI API Key", type="password")
    client = _get_client(st.session_state.openai_key_source, own_key)

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
                recipient, st.session_state.last_report_html or "",
                st.session_state.last_report_subject, creds,
            )
            (st.sidebar.success if ok else st.sidebar.error)(msg)

    return client, user_id, int(arxiv_limit), int(smol_limit), int(max_evaluate)


def main() -> None:
    st.set_page_config(page_title="SignalScout", page_icon="📡", layout="wide")
    inject_theme()
    for k, v in _DEFAULTS.items():
        st.session_state.setdefault(k, v)

    storage = Storage()
    storage.init_db()
    st.session_state["storage"] = storage

    st.title("📡 SignalScout")
    st.caption("Your taste, on autopilot.")

    client, user_id, arxiv_limit, smol_limit, max_evaluate = _render_sidebar(storage)

    bandit = None
    if user_id:
        bandit = TasteBandit(
            user_id=user_id,
            evaluator_ids=[e.id for e in EVALUATORS],
            storage=storage,
        )

    run_tab, arena_tab, learning_tab = st.tabs(["Run", "Arena", "Learning"])
    with run_tab:
        brief_view.render(
            client=client, storage=storage, user_id=user_id,
            arxiv_limit=arxiv_limit, smol_limit=smol_limit, max_evaluate=max_evaluate,
        )
    with arena_tab:
        arena_view.render(storage=storage, user_id=user_id)
    with learning_tab:
        learning_view.render(storage=storage, bandit=bandit, user_id=user_id, client=client)


main()
