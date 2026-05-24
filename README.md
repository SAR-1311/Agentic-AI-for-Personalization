# Agentic AI for Personalization

> An agentic memory framework for long-term personalized AI assistants.
> MSc Informatics dissertation — University of Edinburgh, 2026.

This system turns a stateless LLM into a **long-term personalized companion** by adding:

- 🧠 A **Reasoning Gatekeeper** that scores each interaction by frequency, confidence, and emotional intensity (`I(m) = α·f(m) + β·c(m) + γ·e(m)`).
- 📚 A **three-tier memory hierarchy**: Active Context → Synthesis Layer → Structured Persona.
- 🗑️ **Controlled Forgetting** — temporal decay + user-driven revocation.
- 📊 An evaluation suite measuring **Precision@K, Retrieval Noise Ratio, and Adaptation Latency** against PersonaMem-v2 and Multi-Session Chat.

📖 **Full setup, implementation, and evaluation guide → [`PROJECT_GUIDE.md`](./PROJECT_GUIDE.md)**

---

## Quick start

```bash
git clone https://github.com/SAR-1311/agentic-memory-ai.git
cd agentic-memory-ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # add your OPENAI_API_KEY
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('vader_lexicon')"

# Run backend + frontend
uvicorn backend.main:app --reload --port 8000        # terminal 1
streamlit run frontend/app.py                        # terminal 2
```

Open `http://localhost:8501` for the chat UI, `http://localhost:8000/docs` for the API.

---

## Architecture

```
User → Gatekeeper(I(m)) → Synthesis → STM/LTM → Forgetting → LLM → Reply
```

See [`PROJECT_GUIDE.md §2`](./PROJECT_GUIDE.md#2-architecture-at-a-glance) for the full diagram.

---

## Repository layout

```
backend/         FastAPI service + agent core
frontend/        Streamlit chat UI
evaluation/      PersonaMem, MSC, Life-Transition benchmarks
tests/           pytest unit tests
data/            datasets & databases (gitignored)
notebooks/       analysis & figure generation
PROJECT_GUIDE.md complete step-by-step build guide
```

---

## Status

| Milestone | Description | Status |
|---|---|---|
| M1 | Lit review + design | ⬜ |
| M2 | STM + Gatekeeper + LTM | ⬜ |
| M3 | Forgetting + retrieval | ⬜ |
| M4 | Evaluation done | ⬜ |
| M5 | Dissertation submitted | ⬜ |

---

## License & Ethics

Academic project. The `Controlled Forgetting` feature implements user-driven revocation
(GDPR-style "right to be forgotten") — see `PROJECT_GUIDE.md §4.6`.

## Author

S2845408 — supervised by Mr. Sohan Seth, tutored by Aurora Constantin.
