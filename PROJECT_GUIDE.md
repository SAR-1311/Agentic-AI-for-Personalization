# Agentic AI for Personalization — Complete Project Guide

> **End-to-end implementation roadmap aligned with the Informatics dissertation proposal (UID: S2845408).**
> Project: An agentic memory framework with a Reasoning Gatekeeper, hierarchical memory and controlled forgetting that turns a stateless LLM into a long-term personalized companion.

---

## Table of Contents

1. [Project Overview & Mapping to Proposal](#1-project-overview--mapping-to-proposal)
2. [Architecture at a Glance](#2-architecture-at-a-glance)
3. [Phase 0 — Prerequisites](#phase-0--prerequisites)
4. [Phase 1 — Environment Setup (VS Code)](#phase-1--environment-setup-vs-code)
5. [Phase 2 — Project Structure & Repo Initialisation](#phase-2--project-structure--repo-initialisation)
6. [Phase 3 — Datasets](#phase-3--datasets)
7. [Phase 4 — Core Implementation](#phase-4--core-implementation)
8. [Phase 5 — Backend API (FastAPI)](#phase-5--backend-api-fastapi)
9. [Phase 6 — Frontend (Streamlit)](#phase-6--frontend-streamlit)
10. [Phase 7 — Evaluation](#phase-7--evaluation)
11. [Phase 8 — Testing](#phase-8--testing)
12. [Phase 9 — Deployment](#phase-9--deployment)
13. [Phase 10 — GitHub Workflow](#phase-10--github-workflow)
14. [Phase 11 — Dissertation Write-up](#phase-11--dissertation-write-up)
15. [Troubleshooting](#troubleshooting)
16. [Timeline Mapped to Proposal Gantt](#timeline-mapped-to-proposal-gantt)

---

## 1. Project Overview & Mapping to Proposal

| Proposal Concept | Implementation Module | File |
|---|---|---|
| Reasoning Gatekeeper (Eq. 1: `I(m) = α·f(m) + β·c(m) + γ·e(m)`) | `Gatekeeper` class | `backend/core/gatekeeper.py` |
| Frequency `f(m)` (Eq. 2) | Counter over interaction history | `backend/core/gatekeeper.py` |
| Confidence `c(m)` | LLM-based + linguistic-cue classifier | `backend/core/gatekeeper.py` |
| Emotional intensity `e(m)` (Eq. 3) | VADER / RoBERTa sentiment | `backend/utils/sentiment.py` |
| Active Context (STM) | In-memory deque buffer | `backend/core/memory.py` |
| Synthesis Layer | LLM trait extraction | `backend/core/synthesis.py` |
| Structured Persona (LTM) | SQLite persona table + ChromaDB embeddings | `backend/storage/persona_db.py`, `backend/storage/vector_db.py` |
| Controlled Forgetting (temporal decay Eq. 7 + revocation) | `ForgettingEngine` | `backend/core/forgetting.py` |
| Importance-weighted recall | Top-K with score boost | `backend/core/memory.py` |
| Response generation | LLM with persona + retrieved memories | `backend/core/agent.py` |
| Precision@K (Eq. 4) | Evaluation metric | `evaluation/metrics.py` |
| Retrieval Noise Ratio (Eq. 5) | Evaluation metric | `evaluation/metrics.py` |
| Adaptation Latency (Eq. 6) | Life-Transition test | `evaluation/life_transition.py` |
| PersonaMem-v2 evaluation | Benchmark runner | `evaluation/personamem_eval.py` |
| MSC long-term testing | Multi-session runner | `evaluation/msc_eval.py` |

---

## 2. Architecture at a Glance

```
                ┌─────────────────────────────┐
                │      User (Streamlit UI)    │
                └──────────────┬──────────────┘
                               │ HTTP
                ┌──────────────▼──────────────┐
                │     FastAPI Backend         │
                └──────────────┬──────────────┘
                               │
                ┌──────────────▼──────────────┐
                │   Reasoning Gatekeeper      │  ← computes I(m) = αf + βc + γe
                └──────┬───────────────┬──────┘
                       │  high signal  │ noise
                       ▼               ▼
              ┌────────────────┐   (discarded)
              │ Synthesis Layer│
              └───────┬────────┘
                      │ extract traits
              ┌───────▼────────┐         ┌─────────────────┐
              │ Active Context │◄───────►│ Structured      │
              │   (STM deque)  │         │ Persona (SQLite)│
              └───────┬────────┘         └────────┬────────┘
                      │                           │
                      └─────────────┬─────────────┘
                                    ▼
                       ┌────────────────────────┐
                       │  Vector Store (Chroma) │  ← embeddings + I(m)
                       └────────────┬───────────┘
                                    │
                       ┌────────────▼───────────┐
                       │  Controlled Forgetting │  ← decay + revocation
                       └────────────┬───────────┘
                                    │
                       ┌────────────▼───────────┐
                       │   Response Generator   │
                       └────────────┬───────────┘
                                    │
                                    ▼
                                User reply
```

---

## Phase 0 — Prerequisites

Install once on your machine (Windows/macOS/Linux):

| Tool | Why | Check |
|---|---|---|
| **Python 3.10+** | Core language | `python --version` |
| **Git** | Version control | `git --version` |
| **VS Code** | IDE | (you have this) |
| **VS Code extensions** | Python, Pylance, Jupyter, GitLens, Docker | Install via extensions panel |
| **OpenAI API key** | Default LLM | https://platform.openai.com (≈$5 credit covers full project) |

**Alternative LLMs (free/local):**
- **Ollama** + `llama3.2` or `mistral` — fully local, no key. `curl -fsSL https://ollama.com/install.sh \| sh && ollama pull llama3.2`
- **Hugging Face Inference API** — free tier with key
- **Anthropic Claude API** — high quality, cheap

The `LLMClient` adapter in `backend/utils/llm_client.py` lets you switch with one env var.

---

## Phase 1 — Environment Setup (VS Code)

### 1.1 Clone your empty repo

```bash
# in your projects folder
git clone https://github.com/SAR-1311/<your-repo-name>.git agentic-memory-ai
cd agentic-memory-ai
code .
```

> If you haven't created the repo yet: go to github.com → **New** → name it `agentic-memory-ai` → keep it private until publication → don't initialise with README (we'll push our own).

### 1.2 Create a virtual environment

In the VS Code integrated terminal (`` Ctrl+` ``):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

VS Code should auto-detect it. If not: `Ctrl+Shift+P` → "Python: Select Interpreter" → pick `.venv`.

### 1.3 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # for linguistic cues
python -m nltk.downloader vader_lexicon   # for sentiment
```

### 1.4 Set up `.env`

Copy `.env.example` to `.env` and fill in:

```env
LLM_PROVIDER=openai            # openai | anthropic | ollama | huggingface
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini       # cheap + capable; or gpt-4o
EMBEDDING_MODEL=text-embedding-3-small

# Forgetting
DECAY_LAMBDA=0.05              # λ in Eq. 7; tune during evaluation
DECAY_INTERVAL_HOURS=24

# Gatekeeper weights (α, β, γ in Eq. 1)
WEIGHT_FREQUENCY=0.3
WEIGHT_CONFIDENCE=0.4
WEIGHT_EMOTION=0.3
IMPORTANCE_THRESHOLD=0.35      # below this, discard as noise

# Storage
CHROMA_PERSIST_DIR=./data/chroma
SQLITE_PATH=./data/persona.db
```

### 1.5 Verify

```bash
python -c "import openai, chromadb, fastapi, streamlit; print('OK')"
```

---

## Phase 2 — Project Structure & Repo Initialisation

The full layout (already created in this scaffold):

```
agentic-memory-ai/
├── PROJECT_GUIDE.md          ← this file
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── backend/
│   ├── main.py               ← FastAPI app entrypoint
│   ├── config.py             ← settings loader
│   ├── core/
│   │   ├── agent.py          ← orchestrator
│   │   ├── gatekeeper.py     ← Reasoning Gatekeeper (Eq. 1)
│   │   ├── memory.py         ← STM + LTM + retrieval
│   │   ├── synthesis.py      ← persona extraction
│   │   └── forgetting.py     ← decay + revocation
│   ├── storage/
│   │   ├── persona_db.py     ← SQLite for structured persona
│   │   └── vector_db.py      ← ChromaDB wrapper
│   ├── utils/
│   │   ├── llm_client.py     ← provider-agnostic adapter
│   │   └── sentiment.py      ← e(m) computation
│   ├── models/
│   │   └── schemas.py        ← Pydantic models
│   └── api/
│       └── routes.py         ← REST endpoints
├── frontend/
│   ├── app.py                ← Streamlit chat UI
│   └── requirements.txt
├── evaluation/
│   ├── metrics.py            ← Precision@K, RNR, AL
│   ├── personamem_eval.py    ← PersonaMem benchmark runner
│   ├── msc_eval.py           ← Multi-Session Chat runner
│   └── life_transition.py    ← synthetic stress test
├── tests/
│   ├── test_gatekeeper.py
│   ├── test_memory.py
│   └── test_forgetting.py
├── data/                     ← datasets & databases (gitignored)
└── notebooks/                ← exploration & figures
```

---

## Phase 3 — Datasets

The proposal calls for **PersonaMem-v2** and **Multi-Session Chat (MSC)**. Here is exactly how to obtain and prepare each.

### 3.1 PersonaMem (long-term persona benchmark)

**Source**: The PersonaMem benchmark was introduced by Jiao et al. and is hosted on Hugging Face / GitHub. Search for the most current release:

- HF Hub: https://huggingface.co/datasets — search "PersonaMem"
- GitHub: search `PersonaMem` or `persona memory benchmark`
- Paper: arXiv (referenced in your proposal as 2503.09876)

**If "v2" is not yet public** at the time you start, use the available v1 release and document the swap in your dissertation — your methodology stays valid.

Download script (`data/download_personamem.py`):

```python
from huggingface_hub import snapshot_download
import os

os.makedirs("data/personamem", exist_ok=True)
snapshot_download(
    repo_id="bowen-upenn/PersonaMem",   # confirm the latest org/name
    repo_type="dataset",
    local_dir="data/personamem",
)
print("Downloaded PersonaMem to data/personamem")
```

**Data schema (typical)** — each item contains:
- `dialogue`: list of turns
- `persona_traits`: ground-truth implicit + explicit traits
- `probe_questions`: targeted Q&A to test trait recall
- `session_id`, `turn_id`

### 3.2 Multi-Session Chat (MSC)

**Source**: Released with ParlAI; mirrored on Hugging Face.

```python
from datasets import load_dataset
ds = load_dataset("nayohan/multi_session_chat")   # or similar mirror
# Alternative: use ParlAI directly
# pip install parlai && parlai display_data --task msc
```

Or via ParlAI:
```bash
pip install parlai
parlai display_data --task msc:Session1Self -dt train --num-examples 3
```

MSC contains 5 sessions per dialogue, days/weeks apart — perfect for testing long-term consistency.

### 3.3 Synthetic Life-Transition set

You'll generate this yourself. See `evaluation/life_transition.py`. Pattern:

```
Session 1–3: User says "I'm vegetarian", "I love pasta", "Going to gym tomorrow"
Session 4 (transition): "Started eating meat again after talking to my doctor"
Session 5–6 (probe): "Suggest a recipe" → agent should NOT suggest vegetarian
```

Measure **Adaptation Latency** = number of turns until agent reflects the change.

### 3.4 Data preparation

Run once after download:
```bash
python evaluation/prepare_data.py
```
This normalises both datasets into a common format: `{user_id, session_id, turn_id, role, text, ground_truth_traits}`.

---

## Phase 4 — Core Implementation

This section describes each module's job. **The actual code lives in the matching files** (see `backend/`).

### 4.1 LLM Client (`backend/utils/llm_client.py`)

Adapter pattern — one `generate(prompt)` and `embed(text)` interface, swappable by `LLM_PROVIDER` env var. Supports OpenAI, Anthropic, Ollama, HuggingFace.

### 4.2 Sentiment / Emotional Intensity (`backend/utils/sentiment.py`)

Implements **Eq. 3**: `e(m) = |Sentiment Score(m)|`.
Uses VADER by default (zero-cost, no GPU); upgrade path to `cardiffnlp/twitter-roberta-base-sentiment-latest` for the dissertation comparison.

### 4.3 Reasoning Gatekeeper (`backend/core/gatekeeper.py`)

Implements **Eq. 1**: `I(m) = α·f(m) + β·c(m) + γ·e(m)`.

For each new user utterance:
1. Extract candidate "memory atoms" (statements of preference / fact / event) using LLM with structured output.
2. For each atom, compute:
   - `f(m)` — count similar atoms in LTM (cosine ≥ 0.85) ÷ total interactions (Eq. 2).
   - `c(m)` — confidence via:
     - **Linguistic cues**: hedges ("maybe", "might"), absolutes ("always", "never") — rule-based score.
     - **LLM judge**: secondary LLM call rating certainty 0–1.
     - Final: weighted average.
   - `e(m)` — VADER compound score, abs value, normalised to [0,1].
3. Score `I(m)`. If `I(m) ≥ IMPORTANCE_THRESHOLD`, send to Synthesis Layer; otherwise discard.

### 4.4 Memory Hierarchy (`backend/core/memory.py`)

Three tiers as proposal specifies:

| Tier | Implementation | Lifetime |
|---|---|---|
| **Active Context** | `collections.deque(maxlen=20)` of last turns | Session-only |
| **Synthesis Layer** | Stateless processor (extracts traits) | per-turn |
| **Structured Persona (LTM)** | SQLite `personas` table + ChromaDB embeddings with `importance` metadata | Persistent |

Retrieval: combine vector similarity with importance weighting:
```
final_score = 0.6 * cosine_sim + 0.4 * I(m)_decayed
```

### 4.5 Synthesis Layer (`backend/core/synthesis.py`)

Takes a high-importance memory atom and:
1. Asks LLM (with structured schema): "Extract any persona traits implied by this statement. Return JSON: {trait_type, value, evidence}."
2. Trait types: `preference`, `dietary`, `occupation`, `health`, `relationship`, `goal`, `dislike`, `routine`.
3. Reconciles with existing persona — if a contradiction is detected, the new trait *overwrites* and the old one is marked `superseded` (this is what enables life-transition adaptation).

### 4.6 Controlled Forgetting (`backend/core/forgetting.py`)

Two modes from the proposal:

**Temporal decay** (Eq. 7): `I_t(m) = I_0(m) · e^(-λt)`
- Background job (or on-startup sweep) iterates LTM, recomputes scores by age in days.
- If `I_t(m) < FORGET_FLOOR` (e.g., 0.05) → soft-delete (kept for audit, not retrieved).

**User-driven revocation**:
- API endpoint `DELETE /memory/{memory_id}` and `DELETE /memory/cluster/{trait_type}`.
- Hard-deletes from both SQLite and ChromaDB.
- Required for GDPR-style "right to be forgotten".

### 4.7 Main Agent (`backend/core/agent.py`)

Orchestrator. On each `chat()` call:
1. Push user message to Active Context.
2. Run **Gatekeeper** → score atoms.
3. If high-signal → **Synthesis** → update **Structured Persona**.
4. Retrieve top-K memories from LTM weighted by importance.
5. Build prompt: `[system + persona_summary + retrieved_memories + active_context + user_message]`.
6. Call **LLM** → response.
7. Return response + diagnostic payload (which memories used, scores).

The diagnostic payload is critical for your dissertation — it makes the gatekeeper inspectable, addressing the "Algorithmic Bias / Agentic Drift" point in §3.5 of your proposal.

---

## Phase 5 — Backend API (FastAPI)

Run locally: `uvicorn backend.main:app --reload --port 8000`

Endpoints (`backend/api/routes.py`):

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | Send message, get reply + diagnostics |
| GET | `/persona/{user_id}` | View structured persona |
| GET | `/memory/{user_id}` | List LTM with scores |
| DELETE | `/memory/{memory_id}` | Revoke a single memory |
| DELETE | `/memory/cluster` | Revoke by trait type |
| POST | `/forgetting/run` | Manually trigger decay sweep |
| GET | `/healthz` | Health check |

OpenAPI docs auto-generated at `http://localhost:8000/docs`.

---

## Phase 6 — Frontend (Streamlit)

Run: `streamlit run frontend/app.py`

Three tabs:
1. **Chat** — conversational UI; shows retrieved memories in a side panel (great screenshots for dissertation).
2. **Persona Inspector** — live view of structured persona with confidence + decay-adjusted importance.
3. **Memory Manager** — table of all LTM entries, sortable by `I(m)`; revoke button per row + cluster.

This satisfies the proposal's requirement for "fully personalized" not just backend.

---

## Phase 7 — Evaluation

Run all benchmarks: `python -m evaluation.run_all`.

### 7.1 Persona Extraction Accuracy (PersonaMem)

For each dialogue:
1. Stream turns through agent.
2. After all turns, ask probe questions.
3. Compare agent's persona JSON to ground-truth traits.
4. Metrics: **MCQ accuracy**, **trait F1**, **trait BLEU**.

### 7.2 Retrieval Quality

For each probe, log retrieved top-K vs. ground-truth relevant memories.
- **Precision@K** (Eq. 4): `relevant ∩ retrieved / K`
- **Recall@K**: `relevant ∩ retrieved / relevant_total`
- **Retrieval Noise Ratio (RNR)** (Eq. 5): `irrelevant_retrieved / total_retrieved`

### 7.3 Long-term Consistency (MSC)

Run 5-session dialogues. Between sessions, simulate time gap (advance internal clock by N days for decay testing).
- **Consistency score** = % probe answers consistent with earlier sessions.

### 7.4 Adaptation Latency (Life-Transition)

Generate scenarios where user reverses a preference at session N.
- **Adaptation Latency (Eq. 6)**: `AL = t_adapt − t_change` in turns.

### 7.5 Forgetting Efficacy

After revocation: query agent with a prompt that *should* surface the deleted memory.
- **Target retrieval rate = 0%**.

### 7.6 Baselines

You must compare against:
1. **Vanilla LLM** (no memory).
2. **Flat-RAG baseline** (every turn embedded, top-K cosine, no gatekeeper).
3. **Your full pipeline**.

This is what makes your dissertation defensible — you need to *show* the gatekeeper helps.

Output: `evaluation/results/` — CSV per metric, plots via `notebooks/evaluation_plots.ipynb`.

---

## Phase 8 — Testing

```bash
pytest tests/ -v
pytest tests/ --cov=backend --cov-report=html
```

Critical tests included:
- `test_gatekeeper.py` — score ranges, threshold filtering, cue detection.
- `test_memory.py` — STM eviction, LTM persist + retrieve, importance weighting.
- `test_forgetting.py` — decay math (Eq. 7), revocation removes from both stores.
- `test_synthesis.py` — trait extraction, contradiction handling.

Aim for ≥70% coverage on `backend/core/`.

---

## Phase 9 — Deployment

You have three good options. **For a dissertation, option A is sufficient.** Options B/C are for showing it off.

### Option A — Local demo (recommended for dissertation defence)

```bash
# Terminal 1
uvicorn backend.main:app --port 8000

# Terminal 2
streamlit run frontend/app.py
```

### Option B — Docker Compose (single command, portable)

```bash
docker compose up --build
```
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:8501`

Files included: `Dockerfile`, `docker-compose.yml`.

### Option C — Cloud (publishable demo)

Free tiers that work well:
- **Hugging Face Spaces** (Streamlit template) — push the frontend + a small backend; free GPU possible.
- **Render.com** — free web service for FastAPI; limit: spins down on idle.
- **Fly.io** — generous free tier for both services.

**Avoid** putting your OpenAI key in a public repo. Use the host's secret manager.

---

## Phase 10 — GitHub Workflow

### 10.1 First push

```bash
git init
git add .
git commit -m "feat: initial agentic memory scaffold"
git branch -M main
git remote add origin https://github.com/SAR-1311/agentic-memory-ai.git
git push -u origin main
```

### 10.2 Branching strategy

```
main           ← always working
├── dev        ← integration branch
│   ├── feat/gatekeeper
│   ├── feat/memory-hierarchy
│   ├── feat/forgetting
│   ├── feat/evaluation
│   └── feat/frontend
```

Workflow: `git checkout -b feat/<thing>` → code → `git commit` → PR into `dev` → squash-merge into `main` after tests pass.

### 10.3 Commits aligned to your milestones

| Milestone (proposal Table 1) | Suggested tag |
|---|---|
| M1: Lit review + design done | `v0.1-design` |
| M2: STM + Gatekeeper + LTM | `v0.2-core` |
| M3: Forgetting + retrieval | `v0.3-forgetting` |
| M4: Evaluation done | `v0.4-eval` |
| M5: Dissertation submitted | `v1.0-final` |

Tag with: `git tag v0.2-core && git push --tags`.

### 10.4 GitHub Actions (optional but classy)

Add `.github/workflows/ci.yml` to run `pytest` on every push. Increases dissertation engineering credibility.

---

## Phase 11 — Dissertation Write-up

Your code must produce **figures and numbers** for the dissertation. Set this up early.

### Required figures

1. **Architecture diagram** — already in your proposal Fig. 1, redraw cleanly with Mermaid or Excalidraw.
2. **Importance score distribution** — histogram from `evaluation/results/importance_distribution.png`.
3. **Precision@K curves** — yours vs flat-RAG baseline, K ∈ {1,3,5,10}.
4. **Adaptation latency comparison** — bar chart per scenario.
5. **Forgetting decay curve** — `I_t(m)` vs `t` for sample memories.
6. **Persona evolution timeline** — Gantt-style of when traits were added/superseded.

The notebook `notebooks/evaluation_plots.ipynb` contains the matplotlib code.

### Suggested dissertation chapters

1. Introduction (lift from proposal §1)
2. Background & Related Work (proposal §2 + deeper review)
3. System Design (proposal §3.1 + actual implementation details)
4. Implementation (this guide's Phases 4–6)
5. Evaluation (Phase 7 results)
6. Discussion (limitations, ethics — proposal §3.4–3.5)
7. Conclusion & Future Work
8. References

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ChromaDB tenant default not found` | Delete `data/chroma/` and restart — it auto-recreates. |
| OpenAI rate limit during evaluation | Add `time.sleep(1)` between calls in `evaluation/`; or batch with `asyncio.gather`. |
| `spacy: en_core_web_sm not found` | `python -m spacy download en_core_web_sm` |
| MSC dataset download fails | Use ParlAI fallback: `parlai display_data --task msc` |
| PersonaMem-v2 not yet released | Use v1; cite the proposal's intent and document version actually used. |
| Streamlit can't reach FastAPI | Both must be running; check `BACKEND_URL` in `frontend/app.py`. |
| Memory keeps growing huge during eval | Trigger forgetting sweep: `curl -X POST localhost:8000/forgetting/run` |
| Gatekeeper threshold rejects everything | Lower `IMPORTANCE_THRESHOLD` to 0.2 in `.env` and re-tune. |

---

## Timeline Mapped to Proposal Gantt

| Week | Phase | Deliverable |
|---|---|---|
| **W1 (June)** | Phase 1–2 | Env set up, repo pushed, scaffold compiling |
| **W2** | Phase 3 + literature catch-up | Datasets downloaded, normalised, **M1** ✓ |
| **W3** | Phase 4.1–4.4 | LLM client, sentiment, gatekeeper, STM/LTM working |
| **W4** | Phase 4.5–4.7 | Synthesis + agent orchestrator, **M2** ✓, **D1** prototype |
| **W5** | Phase 4.6 + Phase 5 | Forgetting integrated, FastAPI live |
| **W6** | Phase 6 | Streamlit frontend, **M3** ✓ |
| **W7 (July)** | Phase 7.1–7.3 | PersonaMem + MSC eval running, **D2** integrated agent |
| **W8** | Phase 7.4–7.6 + Phase 8 | Adaptation, forgetting efficacy, baselines, tests, **M4** ✓ |
| **W9 (August)** | Phase 9 + figures | Deployment demo, all evaluation plots, **D3** results report |
| **W10** | Phase 11 | Dissertation polished, **D4 + M5** ✓ |

---

## What to do right now

1. ✅ Read this guide top to bottom.
2. ✅ Complete **Phase 1** (env setup) — should take <30 min.
3. ✅ Push the scaffold to your GitHub.
4. 🎯 Work through **Phase 4.1–4.4** this week — that's your M2 milestone.

Good luck. The proposal design is genuinely strong — execution is now the game.
