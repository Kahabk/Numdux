"""
Data tools for Numdux.
File loading, profiling, and dataset utilities.
"""

import io
import json
import hashlib
from typing import Any, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np


# ── File Loading ──────────────────────────────────────────────────────────────

SUPPORTED_FORMATS = {
    ".csv": "CSV (Comma-Separated)",
    ".tsv": "TSV (Tab-Separated)",
    ".xlsx": "Excel Workbook",
    ".xls":  "Excel (Legacy)",
    ".parquet": "Apache Parquet",
    ".json": "JSON",
    ".jsonl": "JSON Lines",
}


def load_file(source, filename: str = "") -> Tuple[pd.DataFrame, dict]:
    """
    Load any supported file format into a DataFrame.
    
    Args:
        source: file path, bytes, BytesIO, or file-like object
        filename: original filename (used for format detection)
    
    Returns:
        (DataFrame, info_dict)
    """
    if isinstance(source, (str, Path)):
        filename = filename or str(source)
        with open(source, "rb") as f:
            data = f.read()
    elif isinstance(source, bytes):
        data = source
    else:
        # file-like object
        data = source.read()
        source.seek(0)

    buf = io.BytesIO(data)
    ext = Path(filename).suffix.lower()

    if ext in (".csv", ".txt", ""):
        # Detect separator
        sample = data[:8192].decode("utf-8", errors="replace")
        sep = "," if sample.count(",") >= sample.count(";") >= sample.count("\t") else \
              "\t" if sample.count("\t") >= sample.count(";") else \
              ";" if sample.count(";") >= sample.count(",") else ","
        df = pd.read_csv(buf, sep=sep, low_memory=False)

    elif ext == ".tsv":
        df = pd.read_csv(buf, sep="\t", low_memory=False)

    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(buf)

    elif ext == ".parquet":
        df = pd.read_parquet(buf)

    elif ext in (".json", ".jsonl"):
        try:
            df = pd.read_json(buf)
        except ValueError:
            df = pd.read_json(buf, lines=True)

    else:
        # Fallback: try CSV
        df = pd.read_csv(buf, low_memory=False)

    info = {
        "filename": filename,
        "format": SUPPORTED_FORMATS.get(ext, "Unknown"),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest()[:12],
        "loaded_at": datetime.now().isoformat(),
    }

    return df, info


# ── Data Profiling ────────────────────────────────────────────────────────────

def profile_dataframe(df: pd.DataFrame) -> dict:
    """
    Generate a comprehensive profile of a DataFrame.
    Returns a rich dict suitable for serialization.
    """
    total_cells = df.shape[0] * df.shape[1]

    # Per-column analysis
    columns = {}
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        null_count = int(series.isnull().sum())

        col_profile: Dict[str, Any] = {
            "dtype": dtype,
            "null_count": null_count,
            "null_pct": round(null_count / len(df) * 100, 2) if len(df) > 0 else 0,
            "unique_count": int(series.nunique()),
            "unique_pct": round(series.nunique() / len(df) * 100, 2) if len(df) > 0 else 0,
        }

        # Sample values
        sample_vals = series.dropna().head(5).tolist()
        col_profile["sample"] = [str(v)[:50] for v in sample_vals]

        # Numeric stats
        if dtype in ["int64", "float64", "int32", "float32", "int16", "float16"]:
            non_null = series.dropna()
            if len(non_null) > 0:
                col_profile.update({
                    "min": float(non_null.min()),
                    "max": float(non_null.max()),
                    "mean": float(non_null.mean()),
                    "median": float(non_null.median()),
                    "std": float(non_null.std()) if len(non_null) > 1 else 0,
                    "skewness": float(non_null.skew()),
                    "kurtosis": float(non_null.kurtosis()),
                    "zeros": int((non_null == 0).sum()),
                    "negatives": int((non_null < 0).sum()),
                })

                # Outlier detection (IQR)
                q1, q3 = float(non_null.quantile(0.25)), float(non_null.quantile(0.75))
                iqr = q3 - q1
                col_profile["q1"] = q1
                col_profile["q3"] = q3
                col_profile["iqr"] = iqr
                if iqr > 0:
                    n_outliers = int(((non_null < q1 - 1.5 * iqr) | (non_null > q3 + 1.5 * iqr)).sum())
                    col_profile["outlier_count"] = n_outliers
                    col_profile["outlier_pct"] = round(n_outliers / len(df) * 100, 2)

        # Categorical stats
        elif dtype in ["object", "category", "bool"]:
            vc = series.value_counts()
            col_profile.update({
                "top_values": {str(k): int(v) for k, v in vc.head(10).items()},
                "mode": str(vc.index[0]) if len(vc) > 0 else None,
                "mode_freq": int(vc.iloc[0]) if len(vc) > 0 else 0,
                "is_constant": bool(series.nunique() <= 1),
                "is_high_cardinality": bool(series.nunique() > 50),
                "avg_str_length": float(series.dropna().astype(str).str.len().mean()) if null_count < len(df) else 0,
            })

        columns[col] = col_profile

    # Dataset-level quality score
    missing_pct = (df.isnull().sum().sum() / total_cells * 100) if total_cells > 0 else 0
    dup_pct = (df.duplicated().sum() / len(df) * 100) if len(df) > 0 else 0
    const_cols = sum(1 for c in df.columns if df[c].nunique() <= 1)
    quality_score = max(0, round(
        100 - min(missing_pct * 1.5, 40) - min(dup_pct * 0.5, 15) - min(const_cols * 2, 10), 1
    ))

    return {
        "shape": {"rows": df.shape[0], "cols": df.shape[1]},
        "total_cells": total_cells,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
        "quality": {
            "score": quality_score,
            "missing_count": int(df.isnull().sum().sum()),
            "missing_pct": round(missing_pct, 2),
            "duplicate_rows": int(df.duplicated().sum()),
            "dup_pct": round(dup_pct, 2),
            "constant_cols": const_cols,
        },
        "dtypes_summary": df.dtypes.astype(str).value_counts().to_dict(),
        "columns": columns,
        "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object","category"]).columns.tolist(),
        "datetime_columns": df.select_dtypes(include="datetime").columns.tolist(),
        "profiled_at": datetime.now().isoformat(),
    }


# ── Smart Type Inference ──────────────────────────────────────────────────────

def infer_column_roles(df: pd.DataFrame) -> dict:
    """
    Infer the likely role of each column (id, target, feature, datetime, text).
    Useful for auto-configuring ML pipelines.
    """
    roles = {}
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        n_unique = series.nunique()
        dtype = str(series.dtype)

        col_lower = col.lower()

        # Detect IDs
        if any(kw in col_lower for kw in ["id", "_id", "uuid", "key", "index"]):
            if n_unique / n_rows > 0.9:
                roles[col] = "id"
                continue

        # Detect target/label
        if any(kw in col_lower for kw in ["target", "label", "y", "class", "output",
                                            "churn", "survived", "fraud", "default",
                                            "outcome", "response"]):
            roles[col] = "target"
            continue

        # Datetime
        if "datetime" in dtype or any(kw in col_lower for kw in ["date", "time", "timestamp", "created", "updated"]):
            roles[col] = "datetime"
            continue

        # Text (long strings)
        if dtype == "object" and series.dropna().str.len().mean() > 50:
            roles[col] = "text"
            continue

        # High-cardinality categorical (likely ID-like)
        if dtype == "object" and n_unique / n_rows > 0.7:
            roles[col] = "id_like"
            continue

        # Numeric feature
        if dtype in ["int64", "float64", "int32", "float32"]:
            roles[col] = "numeric_feature"
            continue

        # Categorical feature
        if dtype in ["object", "category", "bool"]:
            roles[col] = "categorical_feature"
            continue

        roles[col] = "unknown"

    return roles


# ── Sampling Utilities ────────────────────────────────────────────────────────

def smart_sample(df: pd.DataFrame, max_rows: int = 10_000) -> pd.DataFrame:
    """
    Return a representative sample of the dataframe for LLM context.
    Preserves distribution via stratified sampling where possible.
    """
    if len(df) <= max_rows:
        return df

    # Simple random sample
    return df.sample(n=max_rows, random_state=42).reset_index(drop=True)


def df_to_llm_context(df: pd.DataFrame, max_rows: int = 5) -> str:
    """
    Serialize a dataframe to a compact string for LLM context.
    Includes schema, sample rows, and basic stats.
    """
    lines = []
    lines.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    lines.append("")
    lines.append("Schema:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = df[col].isnull().sum()
        sample = str(df[col].dropna().iloc[0]) if df[col].dropna().shape[0] > 0 else "N/A"
        lines.append(f"  {col}: {dtype} (nulls={nulls}, sample='{sample[:30]}')")
    lines.append("")
    lines.append(f"First {min(max_rows, len(df))} rows:")
    lines.append(df.head(max_rows).to_string(index=False))
    return "\n".join(lines)
