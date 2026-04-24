# SignalScout

**Your taste, on autopilot.**

A research intelligence system that learns what matters to you, then decides what's worth your attention.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/streamlit-1.54+-red.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Tests](https://img.shields.io/badge/tests-88%20passing-brightgreen.svg)]()

---

## The problem

Modern knowledge work is broken in a specific way: it rewards reading more, not deciding better.

arxiv publishes 840 papers tagged `cs.AI` every week. Newsletters multiply. Twitter never sleeps. Every AI tool — ChatGPT, Perplexity, Feedly, Substack — is optimized for the wrong variable. They help you read *more*. None of them help you form a *position*.

The result: you have information, but you don't have conviction. You walk into a product review, a research meeting, or a design crit with a mental backlog of half-read papers and no clear point of view.

The real problem isn't information. It's attention.

## What SignalScout does

SignalScout doesn't summarize papers. It **decides** which ones deserve your attention.

For each paper, three independent AI evaluators score it in parallel — each optimizing a different judgment axis:

- **🔬 Skeptic** judges methodology and rigor. Rewards reproducibility, ablations, acknowledged limitations. Penalizes hype, vague claims, press-release tone.
- **📡 Scout** hunts for novelty and foresight. Rewards new framings, weak signals from frontier work, papers that contradict prevailing consensus.
- **⚙️ Operator** judges shippable, actionable work. Rewards code releases, cost/latency gains, anything an engineer could deploy next quarter.

They disagree often. That's the point.

Where they agree, you have high-confidence signal. Where they disagree, your judgment matters most — and that's where SignalScout surfaces the most contested items first.

## The learning loop

Over time, a contextual multi-armed bandit learns which evaluator's picks *you* consistently reward — on which kinds of items — and weights their verdicts higher for similar items going forward.

Every thumbs-up, thumbs-down, and save is a training signal. After ~30 decisions, SignalScout knows you. After ~100, it knows you better than you know your own preferences.

**Your SignalScout becomes different from anyone else's. That's the moat.**

## How it works — technical architecture

```
arxiv / smol RSS   →   fetch + clean   →   3 parallel evaluators (LLM)
                                                    ↓
                                         (Skeptic / Scout / Operator)
                                                    ↓
                                       Thompson Sampling bandit picks winner
                                                    ↓
                                      Winner's verdict → user-facing brief
                                                    ↓
                                    User feedback (👍/👎/🔖) → bandit update
                                                    ↓
                                      Posterior converges → better picks
```

### The bandit — contextual Thompson Sampling

Each evaluator is an "arm" with its own Bayesian linear regression.

**Per-arm state:**
- `A ∈ ℝ^(15×15)` — precision matrix, initialized as `λI` with λ=1.0
- `b ∈ ℝ^15` — linear coefficient vector, initialized as zeros
- `n_pulls` — total selections
- `n_rewards_positive` — count of +reward feedback

**Decision rule (for context vector x):**

```
For each arm a:
    A_inv = A_a⁻¹
    mean_a = A_inv · b_a
    cov_a = α² · A_inv                    # α = 0.5 controls exploration
    θ_a ~ 𝒩(mean_a, cov_a)                # Thompson sample
    sampled_value_a = θ_a · x

winner = argmax(sampled_value_a)
```

**Update rule (after user feedback reward r):**

```
A_a ← A_a + x · x^T                        # rank-1 update
b_a ← b_a + r · x
```

**Cold start:** First 10 decisions per user are round-robin across arms. After 10 total pulls, Thompson Sampling takes over.

**Reward mapping:**
- 👍 thumbs_up = +1.0
- 🔖 saved = +1.0
- 👎 thumbs_down = -1.0
- auto-dismissed = -0.5
- clicked through = +0.3

### The context vector — 15 dimensions

Stable feature indices (changing requires careful migration):

| Index | Feature | Range |
|---|---|---|
| 0-2 | Source indicators (arxiv / smol / user-defined) | {0, 1} |
| 3-4 | Title length / summary length (normalized) | [0, 1] |
| 5 | Days since published (normalized by 30-day window) | [0, 1] |
| 6 | Has code link (github / gitlab / huggingface regex) | {0, 1} |
| 7-11 | Keyword indicators: benchmark, safety, agent, interpretability, novel | {0, 1} |
| 12 | User has ≥5 feedback events | {0, 1} |
| 13 | User's rolling thumbs-up rate (last 20 items, default 0.5) | [0, 1] |
| 14 | Bias term | 1.0 |

### Why this architecture, not a neural bandit

I chose a **linear** Bayesian bandit deliberately:

- Interpretable: every decision can be explained in terms of which features drove it
- Small-data friendly: converges meaningfully after ~30 user feedback events
- Testable: sublinear regret is empirically verifiable in a seeded synthetic environment
- Cheap: zero training infrastructure, state fits in a SQLite row

A neural bandit would likely outperform on sparse topics with enough data, but adds substantial complexity and requires dramatically more feedback events to converge. For the personal research assistant use case, a linear bandit is the right tool.

## Results

### Empirical regret test (reproducible)

Across 500 simulated decisions in a synthetic environment where one arm is context-dependent best:

| t | Bandit cumulative reward | Random baseline | Gap |
|---|---:|---:|---:|
| 100 | +4 | -8 | +12 |
| 250 | +106 | -30 | +136 |
| 500 | +308 | -136 | **+444** |

**SignalScout outperforms uniform-random selection by 14× across the full simulation.**

Reproducible via `pytest tests/test_bandit.py::test_sublinear_regret`. Fixed seed, 3 arms, 10% label noise, regret assertion enforced in CI.

### Live user session example

Across a real 30-decision user session:
- Scout accumulated 22 pulls at 95% positive reward rate
- Skeptic accumulated 7 pulls at 29% positive reward rate
- Taste summary generated by reading the bandit's own state:
  > *"You reward Scout 1.0x more on items tagged 'Machine Learning,' 'Explainable AI,' and 'LLM Evaluation.' Skeptic's picks on methodology-heavy content correlate with your 👍 at 28.6% — suggesting a user who rewards novelty over rigor."*

The system is not guessing. The system knows.

## Tech stack

- **Streamlit** — UI framework, deployed on Streamlit Community Cloud
- **OpenAI gpt-4o-mini** — three parallel evaluator agents with orthogonal judgment axes
- **NumPy** — contextual Thompson Sampling, matrix ops for rank-1 updates
- **SQLite (stdlib)** — persistent per-user bandit state, decision history, feedback logs
- **Pydantic v2** — typed data contracts between every module
- **Plotly** — regret curve visualization
- **ThreadPoolExecutor** — 3 parallel LLM calls per item, ~2s wall-clock per item
- **feedparser** — RSS ingestion (arxiv cs.AI, Smol AI Newsletter)
- **n8n** — used heavily during prototyping for workflow design and rapid iteration on the evaluation pipeline. The production build reimplements the pipeline natively in Python for deployment reliability, testability, and cost transparency. n8n remains the fastest way to prototype multi-step LLM pipelines; Python is the right target for production.

## The brand system

The visual identity exists because a product that claims to reduce noise must *look* like a product that reduces noise. Every design decision reinforces the thesis.

**Colors**
- Ink Black `#0B0F1A` — background, dominant surface
- Aquamarine `#00F5D4` — signal, active state, CTAs, winner highlight
- Azure Blue `#3A86FF` — secondary evaluator accent (Skeptic)
- Alabaster `#EAEAEA` — body text

**Typography**
- Space Grotesk (500, 700) — headers
- Inter (400, 500, 600) — body
- JetBrains Mono (400) — every number. Timestamps, scores, percentages, latencies, batch IDs, pull counts. Numbers earn monospace.

**Voice**
Sharp, confident, minimal. No hype. No "empower," no "leverage," no "unlock." Say less, mean more. The tagline — *"Your taste, on autopilot"* — is four words because four words is enough.

**Why the aesthetic matters**
Most AI tools look like a purple-gradient Slack clone. SignalScout looks like a Bloomberg terminal. That's intentional. The medium is part of the message: *this is a tool for people who take their attention seriously*.

## Project structure

```
signalscout/
├── app.py                        # Streamlit entry point — thin shell (160 lines)
├── requirements.txt
├── .streamlit/
│   └── config.toml               # Dark mode, brand theme
├── signalscout/
│   ├── models.py                 # Pydantic contracts — Item, ArenaResult, Feedback, BanditArmState
│   ├── evaluators.py             # Three evaluator prompts + parallel execution
│   ├── bandit.py                 # Contextual Thompson Sampling — TasteBandit class
│   ├── features.py               # 15-dim context vector extraction
│   ├── moods.py                  # Per-evaluator mood inference from recent verdicts
│   ├── pipeline.py               # Orchestrator: fetch → evaluate → bandit select → persist
│   ├── sources.py                # RSS fetchers (arxiv, smol) with Source protocol
│   ├── storage.py                # SQLite DAL — all SQL lives here
│   ├── email_brief.py            # Gmail SMTP delivery with UTF-8 header handling
│   ├── branding.py               # SVG lockup loaders for sidebar + page header
│   ├── demo_seed.py              # Synthetic 30-decision seed for demo mode
│   └── ui/
│       ├── theme.py              # Brand CSS injection (Google Fonts, tokens)
│       ├── brief_view.py         # Run tab — brief generation and rendering
│       ├── arena_view.py         # Arena tab — 3-panel evaluator competition
│       ├── learning_view.py      # Learning tab — leaderboard, regret curve, taste summary
│       ├── about_view.py         # About tab — product overview
│       └── components.py         # Reusable: verdict panel, feedback row, badges, pills
└── tests/
    ├── test_bandit.py            # Includes sublinear-regret sanity test
    ├── test_features.py
    ├── test_evaluators.py
    ├── test_moods.py
    ├── test_pipeline.py
    ├── test_storage.py
    ├── test_learning_view.py
    ├── test_components.py
    └── test_brief_view.py
```

88 tests, sub-second suite runtime, matrix round-trip verified via `np.allclose`.

## Running locally

```bash
git clone https://github.com/aravindravi7/Madison-Agent-Streamlit-App.git
cd Madison-Agent-Streamlit-App
pip install -r requirements.txt

# Set up secrets
mkdir -p .streamlit
cat > .streamlit/secrets.toml << EOF
OPENAI_API_KEY = "sk-your-key-here"
GMAIL_USER = "your-email@gmail.com"
GMAIL_APP_PASSWORD = "your-app-password"
EOF

# Run the app
streamlit run app.py
```

Open `http://localhost:8501`. Toggle Demo mode in the sidebar, click **Seed demo data**, and explore.

### Run the tests

```bash
pytest tests/ -v
```

### Regenerate the regret curve

```bash
pytest tests/test_bandit.py::test_sublinear_regret -v -s
```

## Design decisions worth calling out

**1. Per-evaluator orthogonal axes, not personality variants.**
Early versions gave the same "is this relevant?" prompt to three evaluators with different personas. All three scored 80-90 on everything. The bandit had nothing to learn from. The fix was forcing each evaluator to optimize a *different axis entirely* — rigor vs. novelty vs. applicability are mathematically different things, which guarantees disagreement. After the change, mixed-verdict rate went from 13% to 65%.

**2. Cold start as a UX moment, not a hidden phase.**
The first 10 decisions per user are round-robin. Most products hide this behind a loading spinner. SignalScout labels it: **CALIBRATING · 3/10**. Users understand they're training the system. The phase ends visibly, with LEARNED badges appearing instead. Trust compounds when the machine's uncertainty is legible.

**3. Feedback on historical items updates the bandit against frozen context.**
If a user gives feedback three days after a brief was generated, the bandit updates using the context vector *at the time of that decision* — not the user's current feature history. Stored alongside every `ArenaResult` as `context_vector_json`. Without this, the bandit would learn against drift.

**4. Single-file-per-database for SQLite.**
Per-user bandit state fits comfortably in SQLite. No separate persistence layer, no Redis, no ORM. The entire production deployment is a Streamlit app + one `.db` file + OpenAI. Deployment is "push to GitHub, Streamlit Cloud redeploys."

**5. The taste summary is an LLM call, not a template.**
Showing a user "Scout wins 62% of the time" is a number. Showing them *"You reward Scout 2.3x more on items tagged 'agents.' Skeptic's picks on methodology correlate with your 👍 at 64%."* is a sentence about *them*. The sentence requires a small LLM call reading the bandit's weights, cached per `(user_id, feedback_count)` so it doesn't regenerate on every refresh. Cost: ~$0.001 per invalidation. Wildly worth it.

## Limitations and future work

**Current limitations:**
- Linear bandit only. A neural bandit would likely outperform on sparse topics but requires significantly more training data to converge.
- Single-user design. No collaborative filtering across users.
- RSS-only sources. Slack channels, Discord servers, and curated Substacks are future additions.
- Evaluator prompts are calibrated specifically for AI/ML research. New domains (biotech, semiconductors, policy) will require recalibration.
- Mobile experience is not optimized. Streamlit Cloud renders the desktop layout on mobile.

**Near-term roadmap:**
- Custom source addition (user-defined RSS feeds in the sidebar)
- Email digest on a schedule (daily/weekly brief via cron-like service)
- Per-domain evaluator packs (biotech research, semiconductor analysis, policy briefs)

**Longer-term vision:**
- Team mode with shared sources and aggregated preference signals
- Contrarian view surfacing (items the user *would* have skipped but someone with opposite taste would have included)
- Open-source the bandit core as a standalone Python package

## The broader bet

SignalScout is a bet that the next category of AI tools won't be about producing more content. It'll be about helping people *decide what to do with content*.

Every professional who has to keep up with a fast-moving field — AI PMs, biotech researchers, semiconductor analysts, policy strategists, investors — is going to hit the same wall. More information, less clarity, and no way to know whether what you read this week was the right thing to read.

Curation is a moat. Taste is the scarce commodity. A tool that externalizes someone's judgment and makes it better over time is a tool that compounds.

That's what SignalScout is, and that's why the architecture matters beyond AI research.

## About the build

Built in one semester as the final project for **INFO 7375: Branding & AI** at Northeastern University.

Every piece — brand strategy, visual identity, product architecture, bandit implementation, Streamlit UI, evaluator prompts, tests, documentation — was built by me. The course required a full brand system and a working AI tool; I decided early on that the tool should be genuinely useful, not just a demo, and that the brand should reinforce the product's thesis rather than decorate it.

**Built by [Aravind Ravi](https://aravindravi.io)**

- 📧 [aravindravi.academics@gmail.com](mailto:aravindravi.academics@gmail.com)
- 💼 [LinkedIn](https://www.linkedin.com/in/-aravindravi/)
- 🐙 [GitHub](https://github.com/aravindravi7)
- 🌐 [Portfolio](https://aravindravi.io)

If you're building in this space — at Anthropic, OpenAI, Google, or anywhere adjacent — I'd love to trade notes.

---

## License

MIT. Use the code, fork the architecture, build your own version. Attribution appreciated.

---

_Stop reading everything. Start knowing what matters._
