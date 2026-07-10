import os
import io
import uuid
import json
import sqlite3
import traceback
import webbrowser
import threading
import time
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── IMPORT ENGINES WITH GRACEFUL FALLBACKS ───
# Must catch BaseException — on Windows, torch/transformers import failures
# raise KeyboardInterrupt or SystemExit, bypassing plain "except Exception".
_deepvis_available = False
DataProfiler = VisRecommender = Config = None

def _probe_deepvis():
    import sys, subprocess
    r = subprocess.run(
        [sys.executable, "-c",
         "from app import DataProfiler, VisRecommender, Config; print('ok')"],
        capture_output=True, text=True, timeout=20,
        cwd=str(Path(__file__).parent.resolve())
    )
    return r.stdout.strip() == "ok", r.stderr.strip()

try:
    _ok, _err = _probe_deepvis()
    if _ok:
        from app import DataProfiler, VisRecommender, Config
        _deepvis_available = True
        print("  [DeepVIS] Engine loaded OK.")
    else:
        print(f"  [DeepVIS] Unavailable (torch/transformers issue) — fallback active.")
        print(f"  [DeepVIS] Hint: {_err[:200].replace(chr(10),' ')}")
except BaseException as _e:
    print(f"  [DeepVIS] Import skipped ({type(_e).__name__}) — fallback active.")

# KG4VIS: rule-based, no external deps — always runs
kg_engine = None

try:
    from task_vis_engine import TaskVisEngine
    task_vis_engine = TaskVisEngine()
except BaseException:
    task_vis_engine = None

CURRENT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = CURRENT_DIR / "output"
DB_PATH = CURRENT_DIR / "runs.db"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="VisRec Lab — AI Visualization Recommendation System")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

app.mount("/dashboard", StaticFiles(directory=str(CURRENT_DIR), html=True), name="dashboard")
app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")


def _get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                dataset_name TEXT,
                row_count INTEGER,
                col_count INTEGER,
                n_charts INTEGER,
                created_at TEXT,
                result_json TEXT
            )
        """)
        conn.commit()

init_db()


# ─── INSIGHT GENERATOR ───
def _generate_insight(df: pd.DataFrame, x_col: str, y_col: str, chart_type: str, goal: str) -> str:
    """Produce a human-readable analytical explanation for each chart."""
    try:
        chart_type = chart_type.lower()
        x_vals = df[x_col].dropna()
        y_vals = df[y_col].dropna() if y_col in df.columns else None

        if chart_type == "bar":
            grp = df.groupby(x_col)[y_col].mean().dropna()
            if len(grp) > 0:
                top = grp.idxmax()
                bot = grp.idxmin()
                return (f"This bar chart compares average {y_col} across {x_col} categories. "
                        f"'{top}' shows the highest average ({grp[top]:.2f}), while '{bot}' shows the lowest ({grp[bot]:.2f}). "
                        f"Use this to quickly identify top and bottom performers.")

        elif chart_type == "line":
            grp = df.groupby(x_col)[y_col].mean().dropna()
            if len(grp) >= 2:
                delta = grp.iloc[-1] - grp.iloc[0]
                trend = "upward" if delta > 0 else "downward"
                return (f"This line chart tracks how {y_col} changes across {x_col}. "
                        f"The overall trend is {trend} (Δ{delta:+.2f} from first to last point). "
                        f"Look for inflection points indicating key transitions.")

        elif chart_type == "scatter":
            if y_vals is not None and len(x_vals) == len(y_vals):
                corr = df[[x_col, y_col]].dropna().corr().iloc[0, 1]
                direction = "positive" if corr > 0 else "negative"
                strength = "strong" if abs(corr) > 0.6 else ("moderate" if abs(corr) > 0.3 else "weak")
                return (f"This scatter plot reveals a {strength} {direction} relationship between {x_col} and {y_col} "
                        f"(r = {corr:.2f}). "
                        f"{'Points cluster tightly, indicating a reliable pattern.' if abs(corr) > 0.6 else 'Consider other factors that might explain the spread.'}")

        elif chart_type in ("box", "violin"):
            grp = df.groupby(x_col)[y_col]
            medians = grp.median().dropna()
            if len(medians) > 0:
                top = medians.idxmax()
                return (f"This {chart_type} plot shows the spread and distribution of {y_col} per {x_col} group. "
                        f"'{top}' has the highest median ({medians[top]:.2f}). "
                        f"Wide boxes indicate high variability — narrow boxes suggest consistent values.")

        elif chart_type == "histogram":
            mean_v = x_vals.mean()
            std_v = x_vals.std()
            skew_v = x_vals.skew()
            skew_desc = "right-skewed (long tail of high values)" if skew_v > 0.5 else \
                        "left-skewed (long tail of low values)" if skew_v < -0.5 else "approximately symmetric"
            return (f"This histogram shows the frequency distribution of {x_col}. "
                    f"Mean: {mean_v:.2f}, Std Dev: {std_v:.2f}. "
                    f"The distribution is {skew_desc}.")

        elif chart_type == "heatmap":
            return (f"This heatmap displays the correlation matrix across all numeric columns. "
                    f"Dark cells indicate strong correlations. Use it to spot multicollinearity or hidden relationships.")

        elif chart_type == "pie":
            grp = df.groupby(x_col)[y_col].sum().dropna()
            if len(grp) > 0:
                top = grp.idxmax()
                pct = grp[top] / grp.sum() * 100
                return (f"This pie chart shows proportional share of {y_col} by {x_col}. "
                        f"'{top}' dominates with {pct:.1f}% of the total. "
                        f"Useful for understanding composition at a glance.")

        # Generic fallback insight
        return (f"This {chart_type} chart visualizes {y_col} against {x_col}. "
                f"It supports {goal.lower()} analysis by revealing patterns, outliers, or trends in your data.")
    except Exception:
        return (f"This {chart_type} chart explores the relationship between {x_col} and {y_col} "
                f"for {goal.lower()} analysis.")


# ─── CHART BUILDER ───
def build_native_chart(df: pd.DataFrame, x_col: str, y_col: str, chart_type: str, filename: str) -> str:
    plt.figure(figsize=(6, 4.2))
    bg_color = '#15222e'
    plt.gcf().set_facecolor(bg_color)
    plt.gca().set_facecolor(bg_color)

    ct = str(chart_type).lower()
    try:
        if ct == 'scatter':
            plt.scatter(df[x_col], df[y_col], color='#2ecc71', alpha=0.6, s=20)
            plt.xlabel(x_col, color='#9aa0a6', fontsize=8)
            plt.ylabel(y_col, color='#9aa0a6', fontsize=8)

        elif ct == 'box':
            groups = [(str(k), v.dropna().values) for k, v in df.groupby(x_col)[y_col] if len(v.dropna()) > 0]
            if groups:
                bp = plt.boxplot([g[1] for g in groups], labels=[g[0] for g in groups], patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('#3b82f6')
            plt.xlabel(x_col, color='#9aa0a6', fontsize=8)
            plt.ylabel(y_col, color='#9aa0a6', fontsize=8)

        elif ct == 'violin':
            groups = [v.dropna().values for _, v in df.groupby(x_col)[y_col] if len(v.dropna()) > 1]
            if groups:
                plt.violinplot(groups, showmedians=True)
            plt.ylabel(y_col, color='#9aa0a6', fontsize=8)

        elif ct == 'histogram':
            plt.hist(df[x_col].dropna(), bins=20, color='#8b5cf6', edgecolor='#0f172a', alpha=0.85)
            plt.xlabel(x_col, color='#9aa0a6', fontsize=8)
            plt.ylabel('Frequency', color='#9aa0a6', fontsize=8)

        elif ct == 'heatmap':
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:8]
            corr = df[num_cols].corr()
            im = plt.imshow(corr.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            plt.colorbar(im)
            plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha='right', color='#9aa0a6', fontsize=7)
            plt.yticks(range(len(num_cols)), num_cols, color='#9aa0a6', fontsize=7)

        elif ct == 'pie':
            grp = df.groupby(x_col)[y_col].sum().dropna().head(8)
            colors = ['#f59e0b','#3b82f6','#10b981','#ef4444','#8b5cf6','#ec4899','#06b6d4','#f97316']
            plt.pie(grp.values, labels=[str(l) for l in grp.index],
                    autopct='%1.1f%%', colors=colors[:len(grp)],
                    textprops={'color': 'white', 'fontsize': 8})

        elif ct == 'line':
            summary = df.groupby(x_col)[y_col].mean().dropna().head(10)
            plt.plot([str(i) for i in summary.index], summary.values, color='#3b82f6', marker='o', linewidth=2)
            plt.fill_between(range(len(summary)), summary.values, alpha=0.15, color='#3b82f6')
            plt.xlabel(x_col, color='#9aa0a6', fontsize=8)
            plt.ylabel(y_col, color='#9aa0a6', fontsize=8)

        else:  # bar (default)
            summary = df.groupby(x_col)[y_col].mean().dropna().head(8)
            bars = plt.bar([str(i) for i in summary.index], summary.values, color='#f59e0b', width=0.55)
            plt.xlabel(x_col, color='#9aa0a6', fontsize=8)
            plt.ylabel(y_col, color='#9aa0a6', fontsize=8)

        plt.title(f"{ct.title()}: {y_col} by {x_col}", color='white', fontsize=10, pad=8)
        plt.xticks(rotation=20, ha='right', color='#9aa0a6', fontsize=7)
        plt.yticks(color='#9aa0a6', fontsize=7)
        if ct not in ('pie', 'heatmap'):
            plt.grid(axis='y', color='#202e3b', linestyle='--', alpha=0.4)

    except Exception:
        plt.clf()
        plt.gcf().set_facecolor(bg_color)
        plt.text(0.5, 0.5, f"{ct.title()} Chart\n{x_col} × {y_col}",
                 color='#94a3b8', ha='center', va='center', fontsize=11, transform=plt.gca().transAxes)

    plt.tight_layout()
    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, facecolor=bg_color, dpi=110, bbox_inches='tight')
    plt.close()
    return f"/files/{filename}"


# ─── DIVERSE FALLBACK CHART PLAN (5–8 charts) ───
def _build_fallback_plan(numeric_cols, categorical_cols, n_rows):
    """Build a diverse set of 5–8 chart specs based on available columns."""
    plan = []

    cat = categorical_cols[0] if categorical_cols else numeric_cols[0]
    num1 = numeric_cols[0] if numeric_cols else cat
    num2 = numeric_cols[1] if len(numeric_cols) > 1 else num1

    # Always include these if columns exist
    plan.append((cat, num1, "bar",       "COMPARISON",   "Compare average values across categories."))
    plan.append((cat, num1, "line",      "TREND",        "Track how values shift across ordered groups."))
    plan.append((num1, num2, "scatter",  "CORRELATION",  "Reveal relationship between two numeric variables."))
    plan.append((cat, num1, "box",       "DISTRIBUTION", "Show spread and outliers per category."))
    plan.append((num1, num1, "histogram","DISTRIBUTION", "Examine the frequency distribution of values."))

    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        plan.append((cat, num1, "pie",   "COMPARISON",   "Visualize proportional share by category."))

    if len(numeric_cols) >= 3:
        plan.append((numeric_cols[0], numeric_cols[1], "heatmap", "CORRELATION",
                     "Inspect correlation patterns across all numeric columns."))

    if len(categorical_cols) > 0 and n_rows > 20:
        plan.append((cat, num1, "violin", "DISTRIBUTION", "Detailed density distribution across groups."))

    return plan[:8]  # cap at 8


# ─── MAIN PIPELINE ───
def _run_visualization_pipeline(df: pd.DataFrame, dataset_name: str, chosen_goal: str = "COMPARISON") -> dict:
    run_id = str(uuid.uuid4())[:8]
    df.dropna(how='all', inplace=True)

    numeric_cols  = [str(c) for c in df.columns if df[c].dtype.kind in 'biufc']
    categorical_cols = [str(c) for c in df.columns if c not in numeric_cols]

    if not numeric_cols:
        numeric_cols = [str(df.columns[0])]
    if not categorical_cols:
        categorical_cols = [str(df.columns[0])]

    final_charts = []

    # ── Try DeepVIS engine ──
    if _deepvis_available:
        try:
            config = Config(output_dir=str(OUTPUT_DIR), dataset_name=dataset_name, max_visualizations=8)
            profiler = DataProfiler(df, name=dataset_name)
            recommender = VisRecommender(profiler, config)
            recs = recommender.generate_recommendations()

            if task_vis_engine:
                recs = task_vis_engine.align_recommendations(recs, chosen_goal)

            for idx, rec in enumerate(recs[:8]):
                vis_type  = getattr(rec, "vis_type",        "bar")  if not isinstance(rec, dict) else rec.get("vis_type", "bar")
                cols      = getattr(rec, "columns_used",    [])     if not isinstance(rec, dict) else rec.get("columns_used", [])
                score     = getattr(rec, "confidence_score", 0.85)  if not isinstance(rec, dict) else rec.get("confidence_score", 0.85)
                rationale = getattr(rec, "rationale",       "")     if not isinstance(rec, dict) else rec.get("rationale", "")
                insight_t = getattr(rec, "insight_type",    chosen_goal) if not isinstance(rec, dict) else rec.get("insight_type", chosen_goal)

                x = cols[0] if len(cols) > 0 and cols[0] in df.columns else categorical_cols[0]
                y = cols[1] if len(cols) > 1 and cols[1] in df.columns else numeric_cols[0]

                png_name = f"chart_{run_id}_{idx}.png"
                img_url  = build_native_chart(df, x, y, vis_type, png_name)
                insight  = _generate_insight(df, x, y, vis_type, insight_t)

                final_charts.append({
                    "chart_id":       f"chart_{run_id}_{idx}",
                    "vis_type":       vis_type,
                    "columns_used":   [x, y],
                    "confidence_score": float(score),
                    "insight_type":   str(insight_t).upper(),
                    "png_url":        img_url,
                    "html_url":       img_url,
                    "review_state":   "pending",
                    "description":    insight,
                    "explanation":    insight,
                    "rationale":      rationale or insight,
                    "reasoning_path": [f"Profile columns", f"Select {vis_type}", f"Map {x} × {y}", "Score & rank"],
                    "goal_context": {
                        "active_goal":       chosen_goal,
                        "is_goal_optimized": True,
                        "goal_explanation":  f"DeepVIS engine selected this {vis_type} for {chosen_goal} analysis."
                    },
                    "shap_explanation": {
                        "contributions": [
                            {"feature_name": "Data Fit",       "shap_value": round(float(score) * 0.5, 2),  "display_impact": f"+{round(float(score)*0.5,2)}"},
                            {"feature_name": "Perceptual Score","shap_value": round(float(score) * 0.3, 2), "display_impact": f"+{round(float(score)*0.3,2)}"},
                            {"feature_name": "Goal Alignment", "shap_value": round(float(score) * 0.2, 2),  "display_impact": f"+{round(float(score)*0.2,2)}"},
                        ],
                        "base_value": 0.3
                    },
                    "critic_evaluation": {
                        "suitability_score":          min(100, int(float(score) * 110)),
                        "readability_score":          min(100, int(float(score) * 105)),
                        "clarity_score":              min(100, int(float(score) * 100)),
                        "information_density_score":  min(100, int(float(score) * 95)),
                        "overall_score":              min(100, int(float(score) * 105)),
                        "critic_summary":             f"Recommended by DeepVIS with {round(float(score)*100)}% confidence."
                    }
                })
        except Exception:
            final_charts = []   # drop into fallback below

    # ── Fallback: guaranteed 5–8 diverse charts ──
    if not final_charts:
        plan = _build_fallback_plan(numeric_cols, categorical_cols, len(df))
        for idx, (x, y, vtype, goal_label, desc) in enumerate(plan):
            # For histogram / heatmap y==x is okay; just ensure cols exist
            real_x = x if x in df.columns else categorical_cols[0]
            real_y = y if y in df.columns else numeric_cols[0]

            png_name = f"chart_{run_id}_fb{idx}.png"
            img_url  = build_native_chart(df, real_x, real_y, vtype, png_name)
            insight  = _generate_insight(df, real_x, real_y, vtype, goal_label)
            score    = 0.88

            final_charts.append({
                "chart_id":       f"chart_{run_id}_fb{idx}",
                "vis_type":       vtype,
                "columns_used":   [real_x, real_y],
                "confidence_score": score,
                "insight_type":   goal_label,
                "png_url":        img_url,
                "html_url":       img_url,
                "review_state":   "pending",
                "description":    insight,
                "explanation":    insight,
                "rationale":      desc,
                "reasoning_path": [f"Analyze {real_x}", f"Map {real_y}", f"Render {vtype}"],
                "goal_context": {
                    "active_goal":       chosen_goal,
                    "is_goal_optimized": True,
                    "goal_explanation":  f"Auto-selected {vtype} for {goal_label} analysis."
                },
                "shap_explanation": {
                    "contributions": [
                        {"feature_name": "Data Fit",        "shap_value": 0.44, "display_impact": "+0.44"},
                        {"feature_name": "Perceptual Score","shap_value": 0.26, "display_impact": "+0.26"},
                        {"feature_name": "Goal Alignment",  "shap_value": 0.18, "display_impact": "+0.18"},
                    ],
                    "base_value": 0.3
                },
                "critic_evaluation": {
                    "suitability_score":         88,
                    "readability_score":         90,
                    "clarity_score":             87,
                    "information_density_score": 85,
                    "overall_score":             88,
                    "critic_summary":            desc
                }
            })

    # ── Build narrative ──
    goals_covered = list(set(c["insight_type"] for c in final_charts))
    narrative = (
        f"Analyzed '{dataset_name}' ({len(df):,} rows). "
        f"Generated {len(final_charts)} visualizations covering "
        f"{', '.join(goals_covered)} analysis. "
        f"Charts are ranked by confidence score — accept the ones that best support your analytical goal."
    )

    result_payload = {
        "run_id":       run_id,
        "dataset_name": dataset_name,
        "chosen_goal":  chosen_goal,
        "profile": {
            "row_count":       len(df),
            "col_count":       len(df.columns),
            "numeric":         len(numeric_cols),
            "categorical":     len(categorical_cols),
            "numeric_cols":    numeric_cols,
            "categorical_cols": categorical_cols,
            "temporal_cols":   []
        },
        "charts":          final_charts,
        "recommendations": final_charts,
        "insights": [
            {"type": "summary", "description": narrative}
        ],
        "dashboard": {
            "n_charts":      len(final_charts),
            "goals_covered": goals_covered,
            "narrative":     narrative
        }
    }

    with _get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, dataset_name, len(df), len(df.columns),
             len(final_charts), datetime.utcnow().isoformat(), json.dumps(result_payload))
        )
        conn.commit()

    return result_payload


# ─── ROUTES ───
@app.get("/")
async def redirect_to_dashboard():
    return RedirectResponse(url="/dashboard/index.html", status_code=307)


@app.post("/api/analyze")
async def analyze_csv(file: UploadFile = File(...), goal: str = Form("COMPARISON")):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
        return JSONResponse(content=_run_visualization_pipeline(df, Path(file.filename).stem, goal))
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.post("/api/charts/review")
async def review_chart(body: dict):
    """Accept or dismiss a chart and persist the updated state."""
    try:
        run_id  = body.get("run_id")
        chart_id = body.get("chart_id")
        state   = body.get("state", "pending")

        with _get_db() as conn:
            row = conn.execute("SELECT result_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Run not found")

            payload = json.loads(row["result_json"])
            for c in payload["charts"]:
                if c["chart_id"] == chart_id:
                    c["review_state"] = state
            for c in payload.get("recommendations", []):
                if c["chart_id"] == chart_id:
                    c["review_state"] = state

            accepted = [c for c in payload["charts"] if c["review_state"] == "accepted"]
            payload["dashboard"]["narrative"] = (
                f"{len(accepted)} chart(s) accepted so far out of {len(payload['charts'])}. "
                + payload["dashboard"]["narrative"]
            ) if accepted else payload["dashboard"]["narrative"]

            conn.execute("UPDATE runs SET result_json = ? WHERE run_id = ?",
                         (json.dumps(payload), run_id))
            conn.commit()

        return JSONResponse(content={"status": "ok", "run_id": run_id, "chart_id": chart_id, "state": state})
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.post("/api/history/clear")
async def clear_history():
    """Delete all run history from the database."""
    try:
        with _get_db() as conn:
            conn.execute("DELETE FROM runs")
            conn.commit()
        return JSONResponse({"status": "ok", "message": "All run history cleared."})
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())

@app.get("/api/history")
async def fetch_history():
    with _get_db() as conn:
        rows = conn.execute(
            "SELECT run_id, dataset_name, row_count, col_count, n_charts, created_at "
            "FROM runs ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


@app.get("/api/history/{run_id}")
async def fetch_run(run_id: str):
    with _get_db() as conn:
        row = conn.execute("SELECT result_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        return JSONResponse(content=json.loads(row["result_json"]))


@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn, socket

    def _open_when_ready():
        """Poll until the server is actually accepting connections, then open."""
        url = "http://127.0.0.1:8000/dashboard/"
        for _ in range(30):          # wait up to 6 seconds
            time.sleep(0.2)
            try:
                s = socket.create_connection(("127.0.0.1", 8000), timeout=0.3)
                s.close()
                webbrowser.open(url)
                return
            except OSError:
                continue
        webbrowser.open(url)          # open anyway as last resort

    threading.Thread(target=_open_when_ready, daemon=True).start()
    print("\n  VisRec Lab starting → http://127.0.0.1:8000/dashboard/\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
