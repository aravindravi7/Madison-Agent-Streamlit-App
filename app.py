"""
Streamlit App — Madison Agent Research Brief (n8n workflow deployment)
Run with: streamlit run app.py

Replicates workflow_v2: Arxiv + Smol RSS → clean → LLM per-item eval → theme synthesis → HTML brief.
"""

import json
import re
import smtplib
from datetime import datetime
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
import feedparser
from openai import OpenAI

# --- Config ---
ARXIV_RSS = "https://export.arxiv.org/rss/cs.AI"
SMOL_RSS = "https://news.smol.ai/rss.xml"
CUTOFF_DATE = datetime(2023, 1, 1)
DEFAULT_ARXIV_LIMIT = 50
DEFAULT_SMOL_LIMIT = 50
DEFAULT_MAX_EVALUATE = 25


def get_openai_client():
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except StreamlitSecretNotFoundError:
        pass
    api_key = api_key or st.session_state.get("openai_api_key")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def get_gmail_credentials():
    """Return (user, app_password) or (None, None) if not configured."""
    try:
        user = st.secrets.get("GMAIL_USER") or st.secrets.get("gmail_user")
        password = st.secrets.get("GMAIL_APP_PASSWORD") or st.secrets.get("gmail_app_password")
        if user and password:
            return (user, password)
    except StreamlitSecretNotFoundError:
        pass
    return (None, None)


def _normalize_unicode_for_email(s):
    """Replace chars that break ASCII codec (e.g. \\xa0 non-breaking space) with safe equivalents."""
    if not s or not isinstance(s, str):
        return s or ""
    return s.replace("\xa0", " ")  # non-breaking space -> space


def send_brief_email(recipient, html_content, subject):
    """Send the research brief as HTML email to recipient. Returns (success, message)."""
    user, password = get_gmail_credentials()
    if not user or not password:
        return False, "Gmail not configured. Add GMAIL_USER and GMAIL_APP_PASSWORD to .streamlit/secrets.toml (use a Gmail App Password)."
    recipient = _normalize_unicode_for_email(recipient or "").strip()
    if not recipient or "@" not in recipient:
        return False, "Enter a valid email address."
    try:
        subject_safe = _normalize_unicode_for_email(subject or "Research Brief")
        html_safe = _normalize_unicode_for_email(html_content or "")
        msg = MIMEMultipart("alternative")
        msg["Subject"] = Header(subject_safe, "utf-8")
        msg["From"] = user
        msg["To"] = recipient
        msg.attach(MIMEText(html_safe, "html", "utf-8"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(user, password)
            server.sendmail(user, recipient, msg.as_string())
        return True, f"Brief sent to {recipient}."
    except Exception as e:
        return False, str(e)


def fetch_arxiv(limit=100):
    """Fetch and normalize Arxiv RSS; limit = max entries to take from feed."""
    feed = feedparser.parse(ARXIV_RSS)
    out = []
    for e in feed.entries[:limit]:
        out.append({
            "source": "Arxiv",
            "title": e.get("title", ""),
            "url": e.get("link", ""),
            "published_at": e.get("published", ""),
            "summary": e.get("summary", "") or (e.get("content", [{}])[0].get("value", "") if e.get("content") else ""),
            "raw": dict(e),
        })
    return out


def fetch_smol(limit=100):
    """Fetch and normalize Smol RSS; limit = max entries to take from feed."""
    feed = feedparser.parse(SMOL_RSS)
    out = []
    for e in feed.entries[:limit]:
        summary = e.get("summary", "") or e.get("content", [{}])[0].get("value", "") if e.get("content") else ""
        # content:encoded if present
        if not summary and "content_encoded" in e:
            summary = e["content_encoded"]
        out.append({
            "source": "Smol",
            "title": e.get("title", ""),
            "url": e.get("link", ""),
            "published_at": e.get("published", ""),
            "summary": summary,
            "raw": dict(e),
        })
    return out


def clean_and_validate(rows):
    """Dedupe by URL, require summary >= 50 chars, published >= 2023-01-01."""
    seen = set()
    cleaned = []
    for r in rows:
        url = r.get("url") or r.get("link")
        if not url:
            continue
        raw = r.get("raw") or {}
        if isinstance(raw, str):
            try:
                raw = json.loads(raw) if raw else {}
            except Exception:
                raw = {}
        summary = (
            r.get("summary")
            or (raw.get("summary") if isinstance(raw.get("summary"), str) else None)
            or (raw.get("description") if isinstance(raw.get("description"), str) else None)
            or ""
        )
        summary = summary or ""
        if isinstance(summary, dict):
            summary = summary.get("value", summary.get("value", "")) or ""
        summary = str(summary or "").strip()
        if not summary or len(summary) <= 50:
            continue
        pub = r.get("published_at") or r.get("isoDate") or r.get("pubDate") or ""
        d = None
        if pub:
            try:
                d = datetime.fromisoformat(str(pub).replace("Z", "+00:00").strip())
            except Exception:
                try:
                    d = parsedate_to_datetime(str(pub))
                except Exception:
                    d = None
        if not d:
            continue
        d_naive = d.replace(tzinfo=None) if d.tzinfo else d
        if d_naive < CUTOFF_DATE:
            continue
        if url in seen:
            continue
        seen.add(url)
        cleaned.append({
            "source": r.get("source", ""),
            "title": r.get("title", ""),
            "url": url,
            "published_at": d.isoformat() if d else "",
            "summary": summary,
            "raw": json.dumps(r.get("raw") or {}),
        })
    return cleaned


def eval_one_item(client, item):
    """Run LLM evaluation for one item; return dict with output fields."""
    prompt = f"""You are an assistant helping build a model behavior evaluation research brief.

You will be given ONE content item (title + summary + source + url + published date). Your job is to:
- Tag the topic
- Score how relevant it is to model behavior evaluation
- Produce a clean, human-readable 1–2 sentence summary
- Decide whether to include it in the brief

Output rules (strict):
- Return a JSON object only. No markdown, no code fences, no extra text.
- Use exactly these keys: topic_tags, eval_relevance_score, why_it_matters, clean_summary, action, action_reason

Scoring (0–100 integer; spread scores):
- 90–100 (include): directly about evaluation methods, benchmarks, eval frameworks, red-teaming, alignment/safety evals.
- 70–89 (include): strong relevance to LLM behavior/capabilities/risks.
- 40–69 (usually skip): AI/ML only indirectly related.
- 0–39 (skip): no meaningful connection to LLM behavior/evaluation.

Action: if score >= 70 → action = "include"; if score < 70 → action = "skip".

topic_tags: 3–6 concise tags (e.g. "LLM Evaluation", "Benchmarking", "Safety").
why_it_matters: 1–2 sentences, specific.
clean_summary: 1–2 sentences for product/engineering audience.
action_reason: 1 sentence explaining include/skip.

Input content item:
Source: {item.get('source', '')}
Title: {item.get('title', '')}
Published: {item.get('published_at', '')}
URL: {item.get('url', '')}
Summary: {item.get('summary', '')}

Return only the JSON object."""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Strip markdown code block if present
        if "```" in text:
            text = re.sub(r"^```\w*\n?", "", text).rstrip("`").strip()
        # Try to extract JSON if there's extra text
        data = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
        if not data:
            raise ValueError("No JSON in response")
        # Normalize keys (model may use different casing)
        key = lambda d, *ks: next((d[k] for k in ks if k in d), None)
        score = int(key(data, "eval_relevance_score", "eval_relevance_Score") or 0)
        action = str(key(data, "action", "Action") or "skip").lower().strip()
        if score >= 70 and action != "include":
            action = "include"
        return {
            **item,
            "output": {
                "topic_tags": data.get("topic_tags", []) or [],
                "eval_relevance_score": score,
                "why_it_matters": (key(data, "why_it_matters", "Why_it_matters") or ""),
                "clean_summary": (key(data, "clean_summary", "Clean_summary") or ""),
                "action": action,
                "action_reason": (key(data, "action_reason", "Action_reason") or ""),
            },
        }
    except Exception as e:
        return {
            **item,
            "output": {
                "topic_tags": [],
                "eval_relevance_score": 0,
                "why_it_matters": "",
                "clean_summary": item.get("summary", "")[:200],
                "action": "skip",
                "action_reason": str(e),
            },
        }


def prepare_theme_input(included_items):
    """Build theme_input and tag_counts from included items."""
    theme_input = []
    tag_counts = {}
    for it in included_items:
        o = it.get("output") or it
        rec = {
            "title": it.get("title") or o.get("title", ""),
            "url": it.get("url") or o.get("url", ""),
            "tags": o.get("topic_tags") or o.get("tags") or [],
            "summary": o.get("clean_summary") or o.get("summary", ""),
            "why_it_matters": o.get("why_it_matters", ""),
            "score": int(o.get("eval_relevance_score") or o.get("score") or 0),
        }
        theme_input.append(rec)
        for t in rec.get("tags") or []:
            k = str(t).strip()
            if k:
                tag_counts[k] = tag_counts.get(k, 0) + 1
    return theme_input, tag_counts


def run_theme_synthesis(client, theme_input, tag_counts):
    """Run theme synthesis LLM; return themes + editorial_insight + meta_observation."""
    prompt = """You are an editorial analyst synthesizing insights across multiple research items.
Detect patterns, recurring themes, and shifts in focus. Do NOT summarize items individually.
Reason across the entire set for higher-level insights.

You are given a batch of included research items (topic_tags, summary, why_it_matters, relevance score).
Goals:
1. Detect recurring themes
2. Identify which themes dominate vs emerging
3. Infer what this batch suggests about current research focus

Return valid JSON only, with these exact keys:
- "themes": array of objects, each with "name" (string), "confidence" (number 0-1), "evidence" (array of short strings)
- "editorial_insight": string (1-2 sentences)
- "meta_observation": string (1 sentence)

INPUT DATA:
"""
    prompt += json.dumps(theme_input, indent=2)
    prompt += "\n\nTAG FREQUENCY (reference only):\n"
    prompt += json.dumps(tag_counts, indent=2)
    prompt += "\n\nReturn only the JSON object."

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text).rstrip("`")
        return json.loads(text)
    except Exception:
        return {
            "themes": [],
            "editorial_insight": "Theme synthesis unavailable.",
            "meta_observation": "",
        }


def build_report_html(container, theme_synthesis=None):
    """Build HTML string for the research brief."""
    title = container.get("report_title", "Madison Agent Research Brief")
    batch_id = container.get("export_batch_id", "")
    generated_at = container.get("generated_at", datetime.utcnow().isoformat() + "Z")
    items = container.get("included_items", [])

    def esc(s):
        s = str(s or "")
        for a, b in [("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"), ('"', "&quot;")]:
            s = s.replace(a, b)
        return s

    def get_score(it):
        o = it.get("output") or it
        return int(o.get("eval_relevance_score") or o.get("score") or 0)

    def get_tags(it):
        o = it.get("output") or it
        t = o.get("topic_tags") or o.get("tags") or []
        return ", ".join(t) if isinstance(t, list) else str(t)

    def get_summary(it):
        o = it.get("output") or it
        return o.get("clean_summary") or o.get("summary", "")

    def get_why(it):
        o = it.get("output") or it
        return o.get("why_it_matters", "")

    def get_reason(it):
        o = it.get("output") or it
        return o.get("action_reason", "")

    sorted_items = sorted(items, key=get_score, reverse=True)
    rows_html = ""
    for idx, it in enumerate(sorted_items):
        rows_html += f"""
        <tr>
          <td style="padding:10px;border-top:1px solid #e5e5e5;vertical-align:top;width:60px;"><b>{esc(get_score(it))}</b></td>
          <td style="padding:10px;border-top:1px solid #e5e5e5;vertical-align:top;">
            <div style="font-size:14px;font-weight:700;margin-bottom:4px;">Item {idx + 1}</div>
            <div style="font-size:12px;color:#444;margin-bottom:8px;"><b>Tags:</b> {esc(get_tags(it))}</div>
            <div style="margin-bottom:8px;line-height:1.45;"><b>Summary:</b> {esc(get_summary(it))}</div>
            <div style="margin-bottom:8px;line-height:1.45;"><b>Why it matters:</b> {esc(get_why(it))}</div>
            <div style="color:#555;"><b>Reason:</b> {esc(get_reason(it))}</div>
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
            f'<li style="margin:6px 0;"><b>{esc(t.get("name",""))}</b> — {esc(t.get("evidence", [""])[0] if t.get("evidence") else "")}</li>'
            for t in dom[:8]
        )
        theme_html = f"""
        <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;margin-top:14px;">
          <tr><td style="background:#fff;border:1px solid #e6e6e6;border-radius:12px;padding:14px;">
            <div style="font-size:16px;font-weight:700;margin-bottom:8px;">Weekly Theme Synthesis</div>
            <ul style="margin:6px 0 0 18px;padding:0;">{theme_list}</ul>
            {f'<div style="margin-top:12px;"><b>Editorial insight:</b> {esc(editorial)}</div>' if editorial else ''}
            {f'<div style="margin-top:10px;color:#555;"><b>Meta observation:</b> {esc(meta)}</div>' if meta else ''}
          </td></tr>
        </table>
        """

    return f"""
    <!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
    <body style="margin:0;font-family:Arial,sans-serif;background:#f6f7f9;color:#111;min-height:100vh;">
    <div style="font-family:Arial,sans-serif;background:#f6f7f9;padding:18px;color:#111;">
      <div style="max-width:900px;margin:0 auto;">
        <div style="background:#fff;border:1px solid #e6e6e6;border-radius:12px;padding:16px;">
          <div style="font-size:20px;font-weight:800;margin-bottom:6px;">{esc(title)}</div>
          <div style="font-size:12px;color:#666;">
            <div><b>Batch:</b> {esc(batch_id)}</div>
            <div><b>Generated:</b> {esc(generated_at)}</div>
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


def render_report_native(container):
    """Render the research brief using native Streamlit components (always viewable)."""
    st.subheader(container.get("report_title", "Madison Agent Research Brief"))
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Batch", container.get("export_batch_id", "—"))
    with col2:
        st.metric("Generated", container.get("generated_at", "—")[:19].replace("T", " "))
    with col3:
        st.metric("Included", container.get("included_count", 0))
    theme = container.get("theme_synthesis") or {}
    if theme:
        with st.expander("Weekly theme synthesis", expanded=True):
            for t in (theme.get("themes") or [])[:8]:
                if isinstance(t, dict):
                    st.markdown(f"**{t.get('name', '')}** — {t.get('evidence', [''])[0] if t.get('evidence') else ''}")
            if theme.get("editorial_insight"):
                st.markdown(f"**Editorial insight:** {theme['editorial_insight']}")
            if theme.get("meta_observation"):
                st.markdown(f"*Meta observation:* {theme['meta_observation']}")
    items = container.get("included_items") or []
    st.markdown("---")
    st.subheader("Included items")
    if not items:
        st.info("No items met the include threshold (score ≥ 70). Try running again or check the evaluated items below.")
    for i, it in enumerate(sorted(items, key=lambda x: (x.get("output") or {}).get("eval_relevance_score", 0), reverse=True)):
        o = it.get("output") or {}
        score = o.get("eval_relevance_score", 0)
        title = it.get("title", "Item")[:80]
        with st.expander(f"Score **{score}** — {title}…"):
            st.markdown(f"**Tags:** {', '.join(o.get('topic_tags') or [])}")
            st.markdown(f"**Summary:** {o.get('clean_summary', '') or it.get('summary', '')[:300]}")
            st.markdown(f"**Why it matters:** {o.get('why_it_matters', '')}")
            st.markdown(f"*{o.get('action_reason', '')}*")
            if it.get("url"):
                st.markdown(f"[Link]({it['url']})")


def run_workflow(client, arxiv_limit, smol_limit, max_evaluate):
    """Execute the full pipeline; return (report_html, container_dict)."""
    # 1. Fetch RSS (with user-configured limits per source)
    arxiv = fetch_arxiv(limit=arxiv_limit)
    smol = fetch_smol(limit=smol_limit)
    combined = arxiv + smol
    cleaned = clean_and_validate(combined)
    limited = cleaned[:max_evaluate]

    if not limited:
        return None, {"error": "No items passed validation.", "included_items": []}

    # 2. Per-item LLM evaluation
    evaluated = []
    progress = st.progress(0, text="Evaluating items...")
    for i, item in enumerate(limited):
        progress.progress((i + 1) / len(limited), text=f"Evaluating item {i + 1}/{len(limited)}")
        evaluated.append(eval_one_item(client, item))
    progress.empty()

    included = [e for e in evaluated if (e.get("output") or {}).get("action") == "include"]
    batch_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    generated_at = datetime.utcnow().isoformat() + "Z"

    # 3. Theme synthesis
    theme_input, tag_counts = prepare_theme_input(included)
    theme_synthesis = run_theme_synthesis(client, theme_input, tag_counts) if included else {}

    # 4. Build container and HTML
    container = {
        "report_title": "Madison Agent Research Brief",
        "export_batch_id": batch_id,
        "generated_at": generated_at,
        "included_count": len(included),
        "evaluated_count": len(evaluated),
        "included_items": included,
        "theme_synthesis": theme_synthesis,
    }
    html = build_report_html(container, theme_synthesis)
    return html, container


# --- Streamlit UI ---
st.set_page_config(
    page_title="Madison Agent Research Brief",
    page_icon="📋",
    layout="wide",
)

st.title("📋 Madison Agent Research Brief")
st.caption("Deployed from n8n workflow: Arxiv + Smol RSS → LLM evaluation → theme synthesis → brief")

# Session state for limits and last report (for email)
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "openai_key_source" not in st.session_state:
    st.session_state.openai_key_source = "default"  # "default" | "own"
if "last_report_html" not in st.session_state:
    st.session_state.last_report_html = None
if "last_report_subject" not in st.session_state:
    st.session_state.last_report_subject = ""

with st.sidebar:
    st.header("Settings")
    key_source = st.radio(
        "OpenAI API Key",
        options=["default", "own"],
        format_func=lambda x: "Use default key (from app config)" if x == "default" else "Use my own API key",
        index=0 if st.session_state.openai_key_source == "default" else 1,
        key="openai_key_source_radio",
        help="Use the default key (for users without their own) or enter your own OpenAI API key.",
    )
    st.session_state.openai_key_source = key_source
    if key_source == "own":
        api_key = st.text_input(
            "Your OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
            key="openai_api_key_input",
            help="Enter your OpenAI API key. Get one at platform.openai.com.",
        )
        if api_key:
            st.session_state.openai_api_key = api_key
    else:
        st.session_state.openai_api_key = ""
        default_key_set = False
        try:
            default_key_set = bool(st.secrets.get("OPENAI_API_KEY"))
        except Exception:
            pass
        if default_key_set:
            st.caption("Using the default key from app configuration.")
        else:
            st.caption("Default key not set. Add **OPENAI_API_KEY** in app secrets (see README), or switch to \"Use my own API key\" and enter one.")

    st.markdown("---")
    st.subheader("Data source limits")
    st.caption("How many items to pull from each feed and how many to evaluate.")
    arxiv_limit = st.number_input(
        "Arxiv (cs.AI) limit",
        min_value=1,
        max_value=150,
        value=DEFAULT_ARXIV_LIMIT,
        step=5,
        help="Max entries to take from the Arxiv RSS feed.",
    )
    smol_limit = st.number_input(
        "Smol RSS limit",
        min_value=1,
        max_value=150,
        value=DEFAULT_SMOL_LIMIT,
        step=5,
        help="Max entries to take from the Smol RSS feed.",
    )
    max_evaluate = st.number_input(
        "Max items to evaluate (up to 320)",
        min_value=1,
        max_value=320,
        value=DEFAULT_MAX_EVALUATE,
        step=1,
        help="After merging and cleaning, how many items to send to the LLM for include/skip.",
    )

    st.markdown("---")
    st.subheader("Send brief by email")
    st.caption("Enter your email and click Send to receive the latest brief.")
    recipient_email = st.text_input(
        "Your email (Gmail or any)",
        placeholder="you@example.com",
        key="recipient_email",
        help="The brief will be sent to this address when you click Send.",
    )
    # Normalize so pasted text with non-breaking space (\\xa0) doesn't cause ASCII encode errors
    if recipient_email and "\xa0" in recipient_email:
        st.session_state.recipient_email = _normalize_unicode_for_email(recipient_email).strip()
    gmail_ok = get_gmail_credentials()[0] is not None
    has_report = bool(st.session_state.last_report_html)
    if not gmail_ok:
        st.caption("Configure Gmail in `.streamlit/secrets.toml` to enable sending: `GMAIL_USER` and `GMAIL_APP_PASSWORD` (use a Gmail App Password).")
    elif not has_report:
        st.caption("Generate a brief first (run the workflow above), then this button will be enabled.")
    send_btn = st.button("Send brief to my email", disabled=not gmail_ok or not has_report)
    if send_btn and recipient_email:
        success, msg = send_brief_email(
            _normalize_unicode_for_email(recipient_email).strip(),
            st.session_state.last_report_html,
            st.session_state.last_report_subject,
        )
        if success:
            st.success(msg)
        else:
            st.error(msg)

    st.markdown("---")
    st.markdown("**Workflow**")
    st.markdown("- Fetches Arxiv + Smol RSS (limits above)")
    st.markdown("- Cleans & validates (since 2023, dedupe)")
    st.markdown("- Evaluates up to N items with GPT-4o-mini")
    st.markdown("- Includes items with score ≥ 70")
    st.markdown("- Theme synthesis → research brief")

if st.button("Run workflow", type="primary"):
    client = get_openai_client()
    if not client:
        st.error("No OpenAI API key. Use the default key (from app config) or choose \"Use my own API key\" in the sidebar and enter one.")
    else:
        with st.spinner("Running pipeline…"):
            report_html, container = run_workflow(client, arxiv_limit, smol_limit, max_evaluate)
        if report_html:
            num_included = container.get("included_count", 0)
            num_evaled = container.get("evaluated_count", 0)
            st.session_state.last_report_html = report_html
            st.session_state.last_report_subject = f"{container.get('report_title', 'Research Brief')} — {container.get('export_batch_id', '')}"
            st.success(f"Done. Evaluated **{num_evaled}** items, **{num_included}** included in the brief.")
            render_report_native(container)
            with st.expander("HTML preview / export"):
                st.components.v1.html(report_html, height=700, scrolling=True)
                st.download_button(
                    "Download report (HTML)",
                    report_html,
                    file_name=f"research_brief_{container.get('export_batch_id', 'report')}.html",
                    mime="text/html",
                )
            st.info("You can send this brief to your email from the sidebar: enter your address and click **Send brief to my email**.")
        else:
            st.warning(container.get("error", "No report generated."))
