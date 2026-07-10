# VisRec — AI Visualization Recommendation System

> A hybrid AI framework that automatically recommends, ranks, and explains the best data visualizations for any CSV dataset — with human-in-the-loop feedback and adaptive learning.

---

## Overview

VisRec is a research-grade visualization recommendation system built for data analysts, researchers, and students who need to quickly identify the most suitable charts for their datasets. Instead of manually guessing which chart to use, VisRec analyzes your data and recommends 5–8 ranked visualizations with clear, insight-based explanations for each.

The system combines two independent recommendation engines — **DeepVIS** (data-driven, XGBoost + Sentence-BERT) and **KG4VIS** (rule-based knowledge graph) — through a weighted hybrid ranking formula. User Accept/Reject decisions are persisted and used to adjust future rankings, making the system adaptive over time.

---

## System Architecture

```
User uploads CSV + selects analytical goal
              ↓
    FastAPI backend (server.py)
              ↓
  M1 — Data Profiler (app.py)
     Column types, statistics
              ↓
    ┌─────────────────────┐
    ↓                     ↓
M2 — DeepVIS        M3 — KG4VIS
XGBoost + BERT      Rule-based graph
    ↓                     ↓
    └──────┬──────────────┘
           ↓
M4 — Hybrid Ranking Engine
  α × DeepVIS + β × KG4VIS + Feedback Δ
           ↓
M5 — Insight Explanation Generator
  Pattern · Trend · Anomaly · Reason
           ↓
  Matplotlib Chart Renderer
           ↓
M6 — Human Feedback Loop
  Accept → +Δ   Reject → −Δ
           ↓
M7 — Evaluation Metrics
  Relevance · Diversity · Acceptance rate
           ↓
    SQLite (runs.db)
           ↓
  Interactive Dashboard (HTML/CSS/JS)
  ┌──────────┬──────────┬──────────┐
  Recommended  Accepted   Rejected
```

---

## Features

| Feature | Description |
|---|---|
| Hybrid recommendation | Combines DeepVIS + KG4VIS engines through weighted score fusion |
| 5–8 diverse charts | Bar, line, scatter, box, histogram, pie, heatmap, violin |
| Insight explanations | Pattern, trend, anomaly, reason, and recommendation per chart |
| Confidence scoring | Each chart ranked by hybrid confidence score |
| Accept / Reject | Human-in-the-loop review for every recommendation |
| Restore rejected | Rejected charts are never deleted — fully recoverable |
| Feedback learning | Accept/Reject decisions adjust future rankings per chart type |
| Evaluation metrics | Relevance score, diversity, acceptance rate, engine comparison |
| Run history | All previous analyses saved and reloadable from sidebar |
| Clear decisions | Reset Accept/Reject without losing chart data |
| Clear run history | Wipe all stored runs from the database |

---

## Tech Stack

### Backend
| Technology | Purpose |
|---|---|
| Python 3.10 | Core language |
| FastAPI | REST API server |
| Pandas | CSV reading and data processing |
| NumPy | Numerical computation and statistics |
| Matplotlib | Chart image rendering (PNG) |
| SQLite | Local storage for runs and feedback |
| XGBoost | Confidence scoring model (DeepVIS engine) |
| Sentence Transformers | Semantic column name understanding (all-MiniLM-L6-v2) |

### Frontend
| Technology | Purpose |
|---|---|
| HTML / CSS / JavaScript | Single-file dashboard UI |
| Fetch API | Browser-to-server communication |

---

## Project Structure

```
AI_Visualization_Project/
├── server.py             # FastAPI backend — main pipeline orchestrator
├── app.py                # DeepVIS engine — data profiler + recommender
├── index.html            # Frontend dashboard
├── xgb_scorer.py         # XGBoost confidence scoring model
├── kg_reasoner.py        # KG4VIS rule-based knowledge graph engine
├── explainer.py          # Insight explanation generator
├── task_vis_engine.py    # Goal alignment module
├── shap_engine.py        # SHAP-style score breakdown
├── shape_engine.py       # Shape/structure analysis
├── critic_engine.py      # Visual quality assessment
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── output/               # Generated chart images (auto-created)
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/VisRec.git
cd VisRec
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv clean_env
clean_env\Scripts\activate        # Windows
source clean_env/bin/activate     # macOS / Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the server**
```bash
python server.py
```

The server starts at `http://127.0.0.1:8000` and opens the dashboard in your browser automatically.

---

## Usage

1. Open `http://127.0.0.1:8000` in your browser
2. Upload any CSV file using the file upload area
3. Select an analytical goal (Comparison, Trend, Correlation, Distribution, etc.)
4. Click **Run Visualization Analysis**
5. Review the ranked chart recommendations
6. **Accept** charts you find useful or **Reject** ones you don't
7. Rejected charts can be **Restored** at any time
8. Use **Clear All Decisions** to reset without losing charts
9. Previous runs are accessible from the sidebar history

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/analyze` | Upload a CSV and run the full pipeline. Form field: `file` (CSV), `goal` (string) |
| `POST` | `/api/charts/review` | Accept or reject a chart. Body: `run_id`, `chart_id`, `state` |
| `POST` | `/api/history/clear` | Delete all stored run history |
| `GET` | `/api/history` | List all previous analysis runs |
| `GET` | `/api/history/{run_id}` | Fetch full result for a specific run |
| `GET` | `/api/feedback` | View current feedback scores per chart type |
| `GET` | `/files/...` | Serve generated chart PNG images |

---

## Hybrid Ranking Formula

```
Final Score = α × DeepVIS_score + β × KG4VIS_score + Feedback_Δ − Diversity_penalty
```

- `α = 0.55` — weight for the DeepVIS data-driven score
- `β = 0.45` — weight for the KG4VIS rule-based score
- `Feedback_Δ` — cumulative delta from user Accept/Reject history (bounded ±0.20)
- `Diversity_penalty` — applied when more than 2 charts of the same type appear

---

## Evaluation Metrics

After every analysis run the system reports:

| Metric | Description |
|---|---|
| Mean relevance score | Average hybrid confidence score across all charts |
| Diversity score | Unique chart types / total charts |
| Acceptance rate | Accepted / total reviewed charts |
| DeepVIS mean score | Average score from the DeepVIS engine alone |
| KG4VIS mean score | Average score from the KG4VIS engine alone |
| Engine delta | DeepVIS score − KG4VIS score (shows which engine leads) |

---

## Research Context

This project was developed as an IEEE conference paper submission:

> **"VisRec: A Hybrid DeepVIS–KG4VIS Framework for Adaptive Visualization Recommendation with Human-in-the-Loop Feedback"**
>
> Dr. S Alex David, M. Jayasri
> Department of Artificial Intelligence and Machine Learning
> Vel Tech Rangarajan Dr. Sagunthala R&D Institute of Science and Technology, Chennai, India

### References
- Li et al., "KG4Vis: A Knowledge Graph-Based Approach for Visualization Recommendation," IEEE TVCG, 2022
- Hu et al., "VizML: A Machine Learning Approach to Visualization Recommendation," CHI, 2019
- Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System," KDD, 2016
- Reimers & Gurevych, "Sentence-BERT," EMNLP, 2019

---

## Notes

- First run may be slow if Sentence Transformers needs to download model weights (`all-MiniLM-L6-v2`)
- If DeepVIS engine fails to load (torch/transformers compatibility), the system falls back to a built-in rule-based recommender automatically — no crash
- CORS is open (`*`) for local development — tighten `allow_origins` in `server.py` before deploying publicly
- Each analysis run is stored in `runs.db` — delete this file to fully reset all history

---

## License

This project is submitted for academic research purposes.
Department of AIML, Vel Tech R&D Institute of Science and Technology, Chennai, India.

---

*Built with Python · FastAPI · Matplotlib · SQLite · HTML/CSS/JavaScript*
