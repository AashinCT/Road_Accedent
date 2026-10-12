from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_FILE = Path(os.getenv("ACCIDENT_DATASET", str(DATA_DIR / "accidents.csv")))
STATIC_DIR = ROOT / "frontend"
app = FastAPI(title="AI-Based Road Accident Prediction", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
model: Pipeline | None = None
model_features: list[str] = []
model_target: str | None = None
model_metrics: dict[str, Any] = {}
LAT_NAMES = {"lat", "latitude", "gps_latitude", "start_lat", "startlatitude"}
LON_NAMES = {"lon", "lng", "longitude", "gps_longitude", "start_lon", "start_lng", "startlongitude"}
SEVERITY_HINTS = ("severity", "accident_severity", "injury_severity", "crash_severity")

def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(DATA_FILE, low_memory=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read dataset CSV: {exc}") from exc

def find_column(columns: list[str], candidates: set[str]) -> str | None:
    normalized = {str(c).strip().lower().replace(" ", "_"): str(c) for c in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None

def json_safe(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value) if not isinstance(value, (str, int, bool, list, dict)) else value

@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

@app.get("/api/health")
def health() -> dict[str, Any]:
    df = load_data()
    return {"status": "ok", "dataset_loaded": not df.empty, "dataset_path": str(DATA_FILE), "rows": int(len(df))}

@app.get("/api/summary")
def summary() -> dict[str, Any]:
    df = load_data()
    if df.empty:
        return {"dataset_loaded": False, "total_records": 0, "mapped_records": 0, "columns": [],
                "severity_column": None, "severity_counts": {},
                "message": "No CSV loaded. Add a verified dataset at data/accidents.csv."}
    lat_col = find_column(list(df.columns), LAT_NAMES)
    lon_col = find_column(list(df.columns), LON_NAMES)
    sev_col = next((c for c in df.columns if any(h in str(c).lower() for h in SEVERITY_HINTS)), None)
    mapped = 0
    if lat_col and lon_col:
        lat = pd.to_numeric(df[lat_col], errors="coerce")
        lon = pd.to_numeric(df[lon_col], errors="coerce")
        mapped = int((lat.between(-90, 90) & lon.between(-180, 180)).sum())
    counts = {}
    if sev_col:
        counts = {str(k): int(v) for k, v in df[sev_col].fillna("Unknown").value_counts().head(12).items()}
    return {"dataset_loaded": True, "total_records": int(len(df)), "mapped_records": mapped,
            "columns": [str(c) for c in df.columns], "latitude_column": lat_col,
            "longitude_column": lon_col, "severity_column": sev_col, "severity_counts": counts,
            "model_trained": model is not None, "model_metrics": model_metrics,
            "message": "Mapped incidents are shown only when the CSV contains valid coordinates."}

@app.get("/api/incidents")
def incidents(limit: int = 3000) -> dict[str, Any]:
    df = load_data()
    if df.empty:
        return {"incidents": [], "total": 0, "mapped": False}
    lat_col = find_column(list(df.columns), LAT_NAMES)
    lon_col = find_column(list(df.columns), LON_NAMES)
    if not lat_col or not lon_col:
        return {"incidents": [], "total": int(len(df)), "mapped": False,
                "message": "This dataset has no recognized latitude/longitude columns. The map cannot plot its rows."}
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    valid = df[lat.between(-90, 90) & lon.between(-180, 180)].head(max(1, min(limit, 10000))).copy()
    records = []
    for idx, row in valid.iterrows():
        props = {str(k): json_safe(v) for k, v in row.items() if k not in (lat_col, lon_col)}
        records.append({"id": str(idx), "lat": float(pd.to_numeric(row[lat_col])),
                        "lon": float(pd.to_numeric(row[lon_col])), "properties": props})
    return {"incidents": records, "total": int(len(df)), "mapped": True}

@app.get("/api/dataset/inspect")
def inspect_dataset() -> dict[str, Any]:
    df = load_data()
    if df.empty:
        return {"loaded": False, "path": str(DATA_FILE), "message": "Place your CSV at data/accidents.csv.",
                "columns": [], "preview": [], "missing_percent": {}}
    missing = (df.isna().mean() * 100).round(2).to_dict()
    return {"loaded": True, "path": str(DATA_FILE), "rows": int(len(df)),
            "columns": [{"name": str(c), "dtype": str(df[c].dtype), "unique": int(df[c].nunique(dropna=True))}
                        for c in df.columns],
            "preview": [{str(k): json_safe(v) for k, v in r.items()} for r in df.head(5).to_dict(orient="records")],
            "missing_percent": {str(k): float(v) for k, v in missing.items()}}

class TrainRequest(BaseModel):
    target_column: str = Field(min_length=1)
    drop_columns: list[str] = Field(default_factory=list)

class PredictRequest(BaseModel):
    features: dict[str, Any]

@app.post("/api/model/train")
def train(req: TrainRequest) -> dict[str, Any]:
    global model, model_features, model_target, model_metrics
    df = load_data()
    if df.empty:
        raise HTTPException(status_code=400, detail="No dataset loaded. Add data/accidents.csv first.")
    if req.target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column not found. Available columns: {list(df.columns)}")
    data = df.drop(columns=[c for c in req.drop_columns if c in df.columns]).copy()
    y = data.pop(req.target_column)
    valid = y.notna()
    data, y = data.loc[valid], y.loc[valid].astype(str)
    if len(data) < 20:
        raise HTTPException(status_code=400, detail="At least 20 labeled rows are required for a basic train/test split.")
    if y.nunique() < 2:
        raise HTTPException(status_code=400, detail="Target must contain at least two classes.")
    data = data.dropna(axis=1, how="all")
    identifier_like = [c for c in data.columns if data[c].nunique(dropna=True) >= 0.98 * len(data)]
    data = data.drop(columns=identifier_like, errors="ignore")
    if data.empty:
        raise HTTPException(status_code=400, detail="No usable feature columns remain after removing identifier-like columns.")
    numeric_cols = data.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in data.columns if c not in numeric_cols]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols)
    ], remainder="drop")
    pipeline = Pipeline([("preprocess", preprocessor),
                         ("classifier", RandomForestClassifier(n_estimators=250, class_weight="balanced",
                                                               random_state=42, n_jobs=-1))])
    stratify = y if y.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.25, random_state=42, stratify=stratify)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        model, model_features, model_target = pipeline, list(data.columns), req.target_column
        model_metrics = {"accuracy": round(float(accuracy_score(y_test, pred)), 4),
                         "macro_f1": round(float(f1_score(y_test, pred, average="macro", zero_division=0)), 4),
                         "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
                         "classes": sorted(y.unique().tolist()), "target": req.target_column,
                         "excluded_identifier_columns": [str(c) for c in identifier_like],
                         "classification_report": report,
                         "warning": "Holdout metrics are a baseline only; check geography, class imbalance, leakage and dataset provenance."}
        return {"status": "trained", "metrics": model_metrics, "feature_columns": model_features}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Training failed: {exc}") from exc

@app.post("/api/model/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=400, detail="Train a model first using POST /api/model/train.")
    row = {col: req.features.get(col, None) for col in model_features}
    try:
        frame = pd.DataFrame([row], columns=model_features)
        result = str(model.predict(frame)[0])
        output: dict[str, Any] = {"prediction": result, "target": model_target,
                                  "interpretation": "This is a model class prediction, not the probability that an accident will occur."}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(frame)[0]
            output["class_scores"] = {str(c): round(float(p), 4) for c, p in zip(model.named_steps["classifier"].classes_, probs)}
        return output
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
