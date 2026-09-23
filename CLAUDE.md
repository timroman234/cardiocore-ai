# CardioCore AI: notes for Claude Code

Educational Streamlit app (EKG signal processing + Claude tutor). Spec: `prd_cardiocore_ai.md`;
design/decisions: `docs/superpowers/specs/2026-09-23-cardiocore-ai-design.md`.

## Commands
- Run: `uv run streamlit run src/app.py` (run from the repo root so `.streamlit/config.toml` is used)
- Tests: `uv run pytest` (`-m "not network"` offline, `-m live` for a real API call)
- Pre-train the classifier: `uv run python src/model.py` (writes `models/rhythm_rf.joblib`)

## Conventions
- Modules import each other by plain name (`from dsp import ...`); pytest adds `src/` to the path.
- Code is deliberately heavily commented (math, DSP rationale, agent logic): keep that standard.
- Claude model comes from `ANTHROPIC_MODEL` (default `claude-sonnet-5`). Do not pass `temperature`
  or `budget_tokens` (rejected by this model family). Structured output goes through
  `client.messages.parse(output_format=<Pydantic class>)`.
- `.env` holds the API key and is gitignored. Tests must never call the API unless marked `live`
  (the app calls `load_dotenv()` on each run, so patch it in tests that need "no key").
- Streamlit caches imported modules: restart the server after editing `src/*.py` (not just reload).
- Layout must stay on one screen: the right panel scrolls internally (`.st-key-agent_panel`),
  the page must not scroll.
