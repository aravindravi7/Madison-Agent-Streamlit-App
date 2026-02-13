# Madison Agent Research Brief (Streamlit)

Streamlit app that runs the n8n-style workflow: Arxiv + Smol RSS → LLM evaluation → theme synthesis → research brief (view in app, download HTML, or email).

## Setup

1. **Create a virtual environment** (recommended):

   ```bash
   cd /Users/aravindravi/MadisonAgent_StreamlitApp
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Secrets** (optional but recommended for API key and Gmail):

   - Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
   - Add your `OPENAI_API_KEY` and, if you want email, `GMAIL_USER` and `GMAIL_APP_PASSWORD`

## Run

From the project directory:

```bash
cd /Users/aravindravi/MadisonAgent_StreamlitApp
source .venv/bin/activate
streamlit run app.py
```

Or with the venv’s Streamlit directly:

```bash
/Users/aravindravi/MadisonAgent_StreamlitApp/.venv/bin/streamlit run /Users/aravindravi/MadisonAgent_StreamlitApp/app.py
```

The app will open in your browser (e.g. http://localhost:8501).
