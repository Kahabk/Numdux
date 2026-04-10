"""
NumDocsX - FastAPI Backend for Numdux
Provides REST endpoints for dataset analysis, training, and report generation.

Run with: uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
import pandas as pd
import io
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(
    title="NumDocsX API",
    description="Backend API for Numdux — AI Data Co-Pilot",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ──────────────────────────────────────────────────
sessions: Dict[str, Dict] = {}


# ── Request / Response Models ────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    session_id: str
    task: Optional[str] = "full_analysis"
    llm_provider: Optional[str] = "ollama"
    llm_model: Optional[str] = "llama3.2"

class TrainRequest(BaseModel):
    session_id: str
    target_column: str
    feature_columns: Optional[List[str]] = None
    task_type: Optional[str] = "auto"

class ReportRequest(BaseModel):
    session_id: str
    format: Optional[str] = "markdown"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_df(file_bytes: bytes, filename: str) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    name = filename.lower()
    if name.endswith(".csv"):
        return pd.read_csv(buf, low_memory=False)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(buf)
    elif name.endswith(".parquet"):
        return pd.read_parquet(buf)
    elif name.endswith(".json"):
        return pd.read_json(buf)
    return pd.read_csv(buf)

def _df_to_json(df: pd.DataFrame) -> dict:
    return {
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "shape": list(df.shape),
        "sample": df.head(5).to_dict(orient="records"),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "NumDocsX API",
        "version": "0.1.0",
        "status": "running",
        "endpoints": ["/upload", "/auto_analyze", "/train", "/report", "/health"],
    }

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset file. Returns session_id and metadata."""
    try:
        content = await file.read()
        df = _load_df(content, file.filename)

        session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Store session
        from utils.report import generate_markdown_report
        from app import extract_metadata
        metadata = extract_metadata(df, file.filename)

        sessions[session_id] = {
            "df_bytes": content,
            "filename": file.filename,
            "df": df,
            "metadata": metadata,
            "models": [],
            "reports": [],
            "code_cells": [],
            "created_at": datetime.now().isoformat(),
        }

        return {
            "session_id": session_id,
            "filename": file.filename,
            "shape": list(df.shape),
            "metadata": metadata,
            "message": f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns",
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auto_analyze")
async def auto_analyze(request: AnalyzeRequest):
    """Run full automated analysis on uploaded dataset."""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Upload a dataset first.")

    sess = sessions[request.session_id]
    df = sess["df"]
    metadata = sess["metadata"]

    try:
        from agents.orchestrator import BroAIOrchestrator
        orch = BroAIOrchestrator(
            provider=request.llm_provider,
            model=request.llm_model,
        )
        result = orch.run(df=df, metadata=metadata, task=request.task)
        sess["code_cells"].extend(result.get("code_cells", []))

        return {
            "session_id": request.session_id,
            "code_cells_generated": len(result.get("code_cells", [])),
            "logs": result.get("logs", []),
            "status": "success",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def train_model(request: TrainRequest):
    """Train ML models on the uploaded dataset."""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    sess = sessions[request.session_id]
    df = sess["df"]

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
        import warnings
        warnings.filterwarnings("ignore")

        target = request.target_column
        if target not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{target}' not found.")

        features = request.feature_columns or [c for c in df.columns if c != target]
        X = df[features].copy()
        y = df[target].copy()

        for col in X.select_dtypes(include="object").columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X = X.fillna(X.median(numeric_only=True))

        is_clf = (y.nunique() <= 20 or y.dtype == object or request.task_type == "classification")
        if y.dtype == object:
            y = LabelEncoder().fit_transform(y.astype(str))
            is_clf = True

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = []
        if is_clf:
            for name, model in [("RandomForest", RandomForestClassifier(100)),
                                 ("LogisticRegression", __import__('sklearn.linear_model', fromlist=['LogisticRegression']).LogisticRegression(max_iter=500))]:
                model.fit(X_train, y_train)
                acc = accuracy_score(y_test, model.predict(X_test))
                results.append({"model": name, "accuracy": round(acc, 4), "task": "classification"})
        else:
            for name, model in [("RandomForest", RandomForestRegressor(100)),
                                 ("Ridge", __import__('sklearn.linear_model', fromlist=['Ridge']).Ridge())]:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                results.append({
                    "model": name,
                    "r2": round(r2_score(y_test, preds), 4),
                    "mae": round(mean_absolute_error(y_test, preds), 4),
                    "task": "regression",
                })

        sess["models"] = results
        return {"session_id": request.session_id, "results": results, "status": "success"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/report")
async def generate_report(request: ReportRequest):
    """Generate an analysis report for the session."""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    sess = sessions[request.session_id]
    df = sess["df"]
    metadata = sess["metadata"]

    try:
        from utils.report import generate_markdown_report, generate_html_report

        if request.format == "html":
            content = generate_html_report(df, metadata, sess.get("models"))
            media_type = "text/html"
        else:
            content = generate_markdown_report(df, metadata, sess.get("models"))
            media_type = "text/markdown"

        return Response(content=content, media_type=media_type)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Get session status and metadata."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    sess = sessions[session_id]
    return {
        "session_id": session_id,
        "filename": sess.get("filename"),
        "shape": list(sess["df"].shape),
        "code_cells": len(sess.get("code_cells", [])),
        "models": len(sess.get("models", [])),
        "created_at": sess.get("created_at"),
    }


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Delete a session."""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "deleted", "session_id": session_id}
