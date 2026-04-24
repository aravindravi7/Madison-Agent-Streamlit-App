"""About tab — a static product overview that doubles as the pitch.

Uses Streamlit's native markdown so typography inherits the theme.
The 'How each tab works' block is wrapped in a bordered card so the
visual rhythm matches the Arena item wrappers on the other tabs.
"""

from __future__ import annotations

import streamlit as st

__all__ = ["render"]


_HEADER_MD = """
# SignalScout

### Your taste, on autopilot.

SignalScout learns what matters to you, then decides what's worth your
attention. It reads the firehose of AI research, has three AI evaluators
argue about what matters, and learns from your feedback which evaluator
to trust for what kind of content.
"""

_WHAT_IT_DOES_MD = """
## What it does

SignalScout turns raw AI research feeds into a weekly research brief.
For each paper, three independent evaluators score it on a different axis:

- **Skeptic** judges methodology and rigor
- **Scout** hunts for novelty and foresight
- **Operator** judges shippable, actionable work

The evaluators disagree often — which is the point. Where they agree,
you have high-confidence signal. Where they disagree, your judgment
matters most.

Over time, a contextual multi-armed bandit (Thompson Sampling) learns
which evaluator's picks *you* consistently reward with 👍, and weights
their verdicts higher for similar items going forward.
"""

_WHO_ITS_FOR_MD = """
## Who it's for

- AI product managers tracking capability, safety, and evaluation trends
- ML engineers who want signal over noise from arxiv and newsletters
- Researchers who need to keep up with a field moving faster than any
  individual can read
- Strategy + innovation teams monitoring emerging AI capability

Built today for AI research. The same architecture extends to any
high-velocity information domain — biotech, semiconductors, policy,
finance.
"""

_TABS_EXPLAINER_MD = """
## How each tab works

**Run** — Configure data source limits, trigger a brief, view the
ranked output with theme synthesis. The HTML brief is exportable and
can be emailed.

**Arena** — Watch all three evaluators score each item side-by-side.
See who won each decision, why, and how strongly they disagreed. Give
👍/👎/🔖 feedback to teach the bandit your preferences.

**Learning** — See the bandit's current state: per-evaluator pull
counts and win rates, a regret curve showing cumulative learning vs
random baseline, and an LLM-generated summary of what SignalScout has
learned about your specific taste.
"""

_TECH_STACK_MD = """
## Tech stack

- **Streamlit** — UI framework
- **Python** — pipeline orchestration, evaluator concurrency via
  ThreadPoolExecutor
- **OpenAI gpt-4o-mini** — three parallel evaluator agents with
  distinct system prompts (judgment axes: rigor / novelty /
  applicability)
- **NumPy** — contextual Thompson Sampling over a 15-dimension
  feature vector (source, content keywords, user history)
- **SQLite** — persistent bandit state, decision history, feedback
  logs
- **Plotly** — regret curve visualization
- **n8n** — workflow setup and prototyping
"""

_BANDIT_WRITEUP_MD = """
## How the bandit works

Each incoming paper is featurized into a 15-dim context vector.
SignalScout maintains one "arm" per evaluator, each with its own
Bayesian linear regression on (context → reward). On every decision,
Thompson sampling draws from each arm's posterior and picks the arm
with the highest predicted reward. Feedback (👍 = +1, 🔖 = +1,
👎 = −1, 🗙 = −0.5, click = +0.3) updates the winning arm's posterior
via a rank-1 update: `A += xxᵀ`, `b += r·x`.

The first 10 decisions per user are round-robin (cold start); after
that, Thompson sampling takes over.

In testing: across 500 simulated decisions, SignalScout beats a
uniform-random baseline by a cumulative reward gap of +444 units.
"""


def render() -> None:
    """Render the About tab."""
    st.markdown(_HEADER_MD)
    st.markdown("---")
    st.markdown(_WHAT_IT_DOES_MD)
    st.markdown("---")
    st.markdown(_WHO_ITS_FOR_MD)
    st.markdown("---")

    # Wrap the tabs-explainer block in a bordered card for visual rhythm
    # consistent with the Arena item wrappers.
    with st.container(border=True):
        st.markdown(_TABS_EXPLAINER_MD)

    st.markdown("---")
    st.markdown(_TECH_STACK_MD)
    st.markdown("---")
    st.markdown(_BANDIT_WRITEUP_MD)
    st.markdown("---")
    st.markdown(
        '<div style="color:#9CA3AF;font-style:italic;font-size:14px;margin-top:8px;">'
        'Built by Aravind Ravi'
        '</div>',
        unsafe_allow_html=True,
    )
