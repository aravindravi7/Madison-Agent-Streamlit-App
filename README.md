# Madison Agent Research Brief

A Streamlit app that builds AI research briefs from **Arxiv (cs.AI)** and **Smol RSS**: fetch feeds → clean & dedupe → LLM evaluation (GPT-4o-mini) → theme synthesis → HTML brief. View in the app, download as HTML, or send by email.

**Live app:** [View on Streamlit Cloud](https://madison-agent-streamlit-app.streamlit.app) — *Replace with your deployed app URL if different.*

---

## What it does

- **Data sources:** Arxiv cs.AI RSS and [Smol RSS](https://news.smol.ai/rss.xml), with configurable limits
- **Pipeline:** Clean and validate entries (since 2023), deduplicate, score each item with the LLM (include/skip), keep items with score ≥ 70, then synthesize themes into a research brief
- **Output:** In-app report, HTML download, and optional email delivery (Gmail)

---

## Run locally

### 1. Clone and install

```bash
git clone https://github.com/aravindravi7/Madison-Agent-Streamlit-App.git
cd Madison-Agent-Streamlit-App
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Secrets (optional)

For API key and email, copy the example and edit:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`:

- **OpenAI:** `OPENAI_API_KEY` — used when you choose “Use default key” in the app. Get a key at [platform.openai.com](https://platform.openai.com/api-keys).
- **Gmail (optional):** `GMAIL_USER` and `GMAIL_APP_PASSWORD` (use a [Gmail App Password](https://myaccount.google.com/apppasswords); 2-Step Verification required) to enable “Send brief by email.”

Do not commit `secrets.toml`; it is in `.gitignore`.

### 3. Launch

```bash
streamlit run app.py
```

The app opens at http://localhost:8501.

---

## App settings (sidebar)

- **OpenAI API key:** Choose **Use default key** (from `secrets.toml`) or **Use my own API key** and enter your key.
- **Data limits:** Arxiv limit, Smol RSS limit, and **Max items to evaluate (up to 320)**. Default evaluation is 25.
- **Send by email:** Enter a recipient and click **Send brief to my email** after a brief has been generated (Gmail must be set in `secrets.toml`).

---

## Deploy on Streamlit Cloud

1. Push this repo to GitHub (e.g. [aravindravi7/Madison-Agent-Streamlit-App](https://github.com/aravindravi7/Madison-Agent-Streamlit-App)).
2. In [Streamlit Cloud](https://streamlit.io/cloud), create a new app from this repo; set **Main file path** to `app.py`.
3. In the app’s **Settings → Secrets**, paste your TOML (same structure as `secrets.toml`), for example:

   ```toml
   OPENAI_API_KEY = "sk-..."
   GMAIL_USER = "your@gmail.com"
   GMAIL_APP_PASSWORD = "your-app-password"
   ```

4. Deploy. Your live app URL will be shown in the dashboard.

---

## Requirements

- Python 3.9+
- `streamlit>=1.28.0`, `feedparser>=6.0.0`, `openai>=1.0.0` (see `requirements.txt`)

---

## Repository

[https://github.com/aravindravi7/Madison-Agent-Streamlit-App](https://github.com/aravindravi7/Madison-Agent-Streamlit-App)
