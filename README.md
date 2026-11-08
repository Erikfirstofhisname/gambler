# Gambler Lens

React + Vite frontend and FastAPI backend that explain **when and why** Iowa Gambling Task players act suboptimally.

## Features

- Mistake & bias overview (trial-block curves, after-loss vs after-win)
- Player profiles with GMM tiers and personal mistake timelines
- Live situation predictor: **SVM**, Specht **PNN**, and **KL-loss probabilistic net**
- **50-fold leave-one-participant-out CV** to pick the best model, then refit on all data
- Tier board from Gaussian Mixture clustering (A / B / C)

## Data

[`igt_real_5000_behavioral_trials.csv`](igt_real_5000_behavioral_trials.csv) — 50 participants × 100 trials.

- Disadvantageous decks: **1–2** (suboptimal)
- Advantageous decks: **3–4**

## Setup

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "from app.train import train_and_save; train_and_save()"
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — Vite proxies `/api` to the FastAPI server.

## API

- `POST /predict` — deck + switch prediction from situation features
- `GET /participants` / `GET /participants/{id}`
- `GET /bias/overview` / `GET /bias/participant/{id}`
- `GET /tiers`
- `GET /models/metrics`
