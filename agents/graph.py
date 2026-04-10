"""
Numdux Agent Graph — LangGraph state machine definition.
Defines the full multi-agent pipeline as a directed graph.

Each node is an agent that:
  1. Reads from shared state
  2. Generates + (optionally) runs code
  3. Updates shared state with results

Graph flow:
  START → router → [validator → cleaner → feature_engineer →
                    analyst → modeler → advisor] → END
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, TypedDict, Literal
from datetime import datetime


# ── State Schema ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Shared state passed between all agents in the graph."""
    # Input
    df_json: str               # JSON-serialized dataframe (small sample)
    metadata: dict             # Full metadata dict
    task: str                  # User's task description
    llm_provider: str
    llm_model: str

    # Pipeline control
    pipeline: List[str]        # Ordered list of agents to run
    current_agent: str
    completed_agents: List[str]

    # Outputs
    code_cells: List[dict]     # Generated code cells
    logs: List[dict]           # Agent activity log
    insights: List[str]        # Accumulated insights
    quality_issues: List[str]  # Issues found by validator
    model_results: List[dict]  # Model performance results

    # Meta
    error: Optional[str]
    finished: bool


# ── Agent Node Functions ──────────────────────────────────────────────────────

def router_node(state: AgentState) -> AgentState:
    """
    Decide which agents to run based on the task.
    Sets state['pipeline'] with ordered agent names.
    """
    task = state.get("task", "").lower()
    metadata = state.get("metadata", {})

    # Determine pipeline based on task keywords
    if any(w in task for w in ["clean", "quality", "fix", "validate"]):
        pipeline = ["validator", "cleaner"]
    elif any(w in task for w in ["predict", "model", "train", "classify", "regress", "churn", "forecast"]):
        pipeline = ["validator", "cleaner", "analyst", "modeler", "advisor"]
    elif any(w in task for w in ["feature", "engineer", "create features"]):
        pipeline = ["validator", "cleaner", "feature_engineer"]
    elif any(w in task for w in ["insight", "business", "recommend", "summarize"]):
        pipeline = ["analyst", "advisor"]
    elif any(w in task for w in ["eda", "explore", "analysis", "distribution"]):
        pipeline = ["validator", "analyst"]
    else:
        # Default: full pipeline
        pipeline = ["validator", "cleaner", "analyst", "advisor"]

    _log(state, "Router", f"Pipeline selected: {' → '.join(pipeline)}", "info")

    return {**state, "pipeline": pipeline, "current_agent": pipeline[0] if pipeline else ""}


def validator_node(state: AgentState) -> AgentState:
    """Data Validator Agent — checks quality and reports issues."""
    _log(state, "Validator", "Checking data quality...", "thinking")

    code = """# ── Data Quality Validation ──────────────────────────────────────────
import pandas as pd
import numpy as np

print("=" * 65)
print("  DATA QUALITY REPORT — Bro AI Validator Agent")
print("=" * 65)

shape_info = f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns"
print(shape_info)
print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
print()

# ── Missing values ────────────────────────────────────────────────────────
missing = df.isnull().sum()
has_missing = missing[missing > 0].sort_values(ascending=False)
total_missing = missing.sum()
print(f"Missing Values: {total_missing:,} ({total_missing / df.size * 100:.2f}% of all cells)")
if len(has_missing) > 0:
    print()
    print(f"{'Column':<30} {'Count':>8} {'Pct':>8}")
    print("-" * 50)
    for col, cnt in has_missing.items():
        pct = cnt / len(df) * 100
        bar = "█" * int(pct / 5)
        print(f"{col:<30} {cnt:>8,} {pct:>7.1f}% {bar}")
else:
    print("  ✅ No missing values!")

# ── Duplicates ────────────────────────────────────────────────────────────
print()
dups = df.duplicated().sum()
print(f"Duplicate Rows: {dups:,} ({dups / len(df) * 100:.2f}%)")
if dups == 0:
    print("  ✅ No duplicates!")

# ── Data types ────────────────────────────────────────────────────────────
print()
print("Column Types Summary:")
dtype_counts = df.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"  {str(dtype):<15} {count} columns")

# ── Numeric outliers (IQR) ────────────────────────────────────────────────
num_cols = df.select_dtypes(include='number').columns.tolist()
outlier_report = {}
for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    if iqr > 0:
        n_outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
        if n_outliers > 0:
            outlier_report[col] = int(n_outliers)

if outlier_report:
    print()
    print("Potential Outliers (IQR method):")
    for col, n in outlier_report.items():
        print(f"  {col:<30} {n:>6,} ({n/len(df)*100:.1f}%)")
else:
    print()
    print("  ✅ No significant outliers detected!")

# ── Constant / near-constant columns ─────────────────────────────────────
const_cols = [col for col in df.columns if df[col].nunique() <= 1]
if const_cols:
    print()
    print(f"⚠  Constant columns (consider dropping): {const_cols}")

# ── High-cardinality categoricals ─────────────────────────────────────────
cat_cols = df.select_dtypes(include='object').columns
high_card = [col for col in cat_cols if df[col].nunique() > 100]
if high_card:
    print()
    print(f"⚠  High-cardinality categoricals: {high_card}")

print()
print("=" * 65)
print("  Validation complete.")
print("=" * 65)
"""

    issues = []
    metadata = state.get("metadata", {})
    quality = metadata.get("quality", {})
    if quality.get("missing_count", 0) > 0:
        issues.append(f"Missing values: {quality['missing_count']:,} ({quality['missing_pct']}%)")
    if quality.get("duplicate_rows", 0) > 0:
        issues.append(f"Duplicate rows: {quality['duplicate_rows']:,}")

    cell = _make_cell("Validator", code)
    _log(state, "Validator", f"Found {len(issues)} quality issues", "success" if not issues else "warning")

    return {
        **state,
        "code_cells": state.get("code_cells", []) + [cell],
        "quality_issues": issues,
        "completed_agents": state.get("completed_agents", []) + ["validator"],
    }


def cleaner_node(state: AgentState) -> AgentState:
    """Data Cleaner Agent — applies smart cleaning strategies."""
    _log(state, "Cleaner", "Applying auto-cleaning pipeline...", "thinking")

    issues = state.get("quality_issues", [])
    has_missing = any("Missing" in i for i in issues)
    has_dups = any("Duplicate" in i for i in issues)

    code = """# ── Smart Data Cleaning Pipeline ─────────────────────────────────────
import pandas as pd
import numpy as np

df_clean = df.copy()
original_shape = df_clean.shape
steps_taken = []

# ── Step 1: Remove fully empty rows & columns ─────────────────────────────
rows_before = len(df_clean)
df_clean.dropna(how='all', axis=0, inplace=True)
df_clean.dropna(how='all', axis=1, inplace=True)
if len(df_clean) < rows_before:
    steps_taken.append(f"Removed {rows_before - len(df_clean)} fully-empty rows")

# ── Step 2: Remove exact duplicates ──────────────────────────────────────
n_dups = df_clean.duplicated().sum()
if n_dups > 0:
    df_clean.drop_duplicates(inplace=True)
    steps_taken.append(f"Removed {n_dups} duplicate rows")

# ── Step 3: Fix dtypes — try numeric conversion for object cols ───────────
for col in df_clean.select_dtypes(include='object').columns:
    converted = pd.to_numeric(df_clean[col], errors='coerce')
    null_ratio_before = df_clean[col].isnull().mean()
    null_ratio_after = converted.isnull().mean()
    # Only convert if we gain ≥80% numeric and don't increase nulls much
    if (converted.notna().mean() > 0.8 and
            null_ratio_after <= null_ratio_before + 0.05):
        df_clean[col] = converted
        steps_taken.append(f"Converted '{col}' object → numeric")

# ── Step 4: Fix datetime columns ─────────────────────────────────────────
for col in df_clean.select_dtypes(include='object').columns:
    sample = df_clean[col].dropna().head(20)
    try:
        pd.to_datetime(sample, infer_datetime_format=True)
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce', infer_datetime_format=True)
        steps_taken.append(f"Converted '{col}' → datetime")
    except Exception:
        pass

# ── Step 5: Impute missing — numeric with median, cat with mode ───────────
num_cols = df_clean.select_dtypes(include='number').columns
for col in num_cols:
    n_null = df_clean[col].isnull().sum()
    if n_null > 0:
        median_val = df_clean[col].median()
        df_clean[col].fillna(median_val, inplace=True)
        steps_taken.append(f"Imputed {n_null} missing in '{col}' (median={median_val:.4g})")

cat_cols = df_clean.select_dtypes(include='object').columns
for col in cat_cols:
    n_null = df_clean[col].isnull().sum()
    if n_null > 0 and len(df_clean[col].mode()) > 0:
        mode_val = df_clean[col].mode()[0]
        df_clean[col].fillna(mode_val, inplace=True)
        steps_taken.append(f"Imputed {n_null} missing in '{col}' (mode='{mode_val}')")

# ── Step 6: Strip whitespace from string columns ──────────────────────────
for col in df_clean.select_dtypes(include='object').columns:
    df_clean[col] = df_clean[col].str.strip()

# ── Step 7: Remove constant columns ──────────────────────────────────────
const_cols = [c for c in df_clean.columns if df_clean[c].nunique() <= 1]
if const_cols:
    df_clean.drop(columns=const_cols, inplace=True)
    steps_taken.append(f"Dropped constant columns: {const_cols}")

# ── Report ────────────────────────────────────────────────────────────────
print("=" * 65)
print("  CLEANING REPORT — Bro AI Cleaner Agent")
print("=" * 65)
print(f"Original : {original_shape[0]:,} rows × {original_shape[1]} cols")
print(f"Cleaned  : {df_clean.shape[0]:,} rows × {df_clean.shape[1]} cols")
print(f"Removed  : {original_shape[0] - df_clean.shape[0]} rows, {original_shape[1] - df_clean.shape[1]} cols")
print()
if steps_taken:
    print(f"Steps Applied ({len(steps_taken)}):")
    for i, step in enumerate(steps_taken, 1):
        print(f"  {i:2}. ✓ {step}")
else:
    print("  ✅ Dataset was already clean — no changes needed.")

print()
print(f"Remaining nulls : {df_clean.isnull().sum().sum()}")
print(f"Final dtypes    : {df_clean.dtypes.value_counts().to_dict()}")
print("=" * 65)

df = df_clean  # ← Use this cleaned dataframe going forward
"""

    cell = _make_cell("Cleaner", code)
    _log(state, "Cleaner", "Cleaning pipeline generated", "success")

    return {
        **state,
        "code_cells": state.get("code_cells", []) + [cell],
        "completed_agents": state.get("completed_agents", []) + ["cleaner"],
    }


def feature_engineer_node(state: AgentState) -> AgentState:
    """Feature Engineer Agent — creates new features."""
    _log(state, "FeatureEng", "Generating feature engineering code...", "thinking")

    metadata = state.get("metadata", {})
    cols = metadata.get("columns", {})
    num_cols = [c for c, v in cols.items() if v.get("dtype") in ["int64","float64","float32","int32"]]
    dt_cols = [c for c, v in cols.items() if "datetime" in v.get("dtype","").lower()]

    code = f"""# ── Feature Engineering ──────────────────────────────────────────────
import pandas as pd
import numpy as np

df_fe = df.copy()
new_features = []

num_cols = {num_cols[:10]}  # Numeric columns detected

# ── Polynomial features for top numerics ─────────────────────────────────
if len(num_cols) >= 2:
    col_a, col_b = num_cols[0], num_cols[1]
    df_fe[f'{{col_a}}_x_{{col_b}}'] = df_fe[col_a] * df_fe[col_b]
    new_features.append(f'{{col_a}}_x_{{col_b}}')

    df_fe[f'{{col_a}}_div_{{col_b}}'] = df_fe[col_a] / (df_fe[col_b] + 1e-9)
    new_features.append(f'{{col_a}}_div_{{col_b}}')

# ── Log transforms for skewed numeric cols ────────────────────────────────
for col in num_cols[:5]:
    if df_fe[col].min() > 0:
        skew = df_fe[col].skew()
        if abs(skew) > 1.0:
            df_fe[f'log_{{col}}'] = np.log1p(df_fe[col])
            new_features.append(f'log_{{col}}')

# ── Datetime features ─────────────────────────────────────────────────────
dt_cols = {dt_cols}
for col in dt_cols:
    if col in df_fe.columns and pd.api.types.is_datetime64_any_dtype(df_fe[col]):
        df_fe[f'{{col}}_year']    = df_fe[col].dt.year
        df_fe[f'{{col}}_month']   = df_fe[col].dt.month
        df_fe[f'{{col}}_dayofweek'] = df_fe[col].dt.dayofweek
        df_fe[f'{{col}}_is_weekend'] = (df_fe[col].dt.dayofweek >= 5).astype(int)
        new_features.extend([f'{{col}}_year', f'{{col}}_month',
                              f'{{col}}_dayofweek', f'{{col}}_is_weekend'])

# ── Z-score normalization (optional) ─────────────────────────────────────
for col in num_cols[:5]:
    std = df_fe[col].std()
    if std > 0:
        df_fe[f'z_{{col}}'] = (df_fe[col] - df_fe[col].mean()) / std
        new_features.append(f'z_{{col}}')

# ── Report ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  FEATURE ENGINEERING REPORT")
print("=" * 60)
print(f"Original features : {{df.shape[1]}}")
print(f"New features added: {{len(new_features)}}")
print(f"Total features    : {{df_fe.shape[1]}}")
print()
if new_features:
    print("New features:")
    for f in new_features:
        print(f"  ✓ {{f}}")
print()
print(f"Final shape: {{df_fe.shape}}")

df = df_fe  # Update working dataframe
"""

    cell = _make_cell("FeatureEng", code)
    _log(state, "FeatureEng", f"Feature engineering code generated", "success")

    return {
        **state,
        "code_cells": state.get("code_cells", []) + [cell],
        "completed_agents": state.get("completed_agents", []) + ["feature_engineer"],
    }


def analyst_node(state: AgentState) -> AgentState:
    """Analyst Agent — statistical analysis and insights."""
    _log(state, "Analyst", "Running statistical analysis...", "thinking")

    code = """# ── Statistical Analysis & EDA ───────────────────────────────────────
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("  STATISTICAL ANALYSIS — Bro AI Analyst Agent")
print("=" * 65)

num_cols = df.select_dtypes(include='number').columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()
dt_cols  = df.select_dtypes(include='datetime').columns.tolist()

print(f"Numeric    : {len(num_cols)} columns")
print(f"Categorical: {len(cat_cols)} columns")
print(f"Datetime   : {len(dt_cols)} columns")
print()

# ── Descriptive statistics ────────────────────────────────────────────────
if num_cols:
    print("── Numeric Statistics ───────────────────────────────────────────")
    stats = df[num_cols].describe()
    # Add skewness and kurtosis
    stats.loc['skewness'] = df[num_cols].skew()
    stats.loc['kurtosis'] = df[num_cols].kurtosis()
    print(stats.round(4).to_string())
    print()

# ── Correlation analysis ──────────────────────────────────────────────────
if len(num_cols) >= 2:
    print("── Top Correlations ─────────────────────────────────────────────")
    corr = df[num_cols].corr()
    pairs = []
    seen = set()
    for col_a in num_cols:
        for col_b in num_cols:
            if col_a != col_b:
                key = frozenset([col_a, col_b])
                if key not in seen:
                    seen.add(key)
                    pairs.append({'A': col_a, 'B': col_b, 'r': corr.loc[col_a, col_b]})
    pairs_df = pd.DataFrame(pairs).assign(abs_r=lambda x: x['r'].abs())
    top_corr = pairs_df.sort_values('abs_r', ascending=False).head(8)
    print(f"{'Column A':<25} {'Column B':<25} {'r':>8}  Type")
    print("-" * 70)
    for _, row in top_corr.iterrows():
        strength = "Strong" if row['abs_r'] > 0.7 else "Moderate" if row['abs_r'] > 0.4 else "Weak"
        direction = "+" if row['r'] > 0 else "-"
        print(f"{row['A']:<25} {row['B']:<25} {row['r']:>8.4f}  {direction}{strength}")
    print()

# ── Categorical analysis ──────────────────────────────────────────────────
if cat_cols:
    print("── Categorical Analysis ─────────────────────────────────────────")
    for col in cat_cols[:6]:
        vc = df[col].value_counts()
        n_unique = df[col].nunique()
        print(f"  {col} ({n_unique} unique):")
        for val, cnt in vc.head(3).items():
            print(f"    {str(val):<25} {cnt:>6,} ({cnt/len(df)*100:>5.1f}%)")
    print()

# ── Outlier detection ─────────────────────────────────────────────────────
if num_cols:
    print("── Outlier Summary (IQR method) ─────────────────────────────────")
    outlier_found = False
    for col in num_cols[:10]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            n_out = ((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).sum()
            if n_out > 0:
                print(f"  {col:<30} {n_out:>6,} outliers ({n_out/len(df)*100:.1f}%)")
                outlier_found = True
    if not outlier_found:
        print("  ✅ No significant outliers detected!")
    print()

print("=" * 65)
print("  Analysis complete.")
print("=" * 65)
"""

    insights = [
        "Statistical analysis complete — check output for correlations and distributions.",
        "Use the EDA tab for interactive visualizations.",
    ]

    cell = _make_cell("Analyst", code)
    _log(state, "Analyst", "Analysis pipeline generated", "success")

    return {
        **state,
        "code_cells": state.get("code_cells", []) + [cell],
        "insights": state.get("insights", []) + insights,
        "completed_agents": state.get("completed_agents", []) + ["analyst"],
    }


def modeler_node(state: AgentState) -> AgentState:
    """Modeler Agent — trains and evaluates ML models."""
    _log(state, "Modeler", "Training ML models...", "thinking")

    metadata = state.get("metadata", {})
    cols = metadata.get("columns", {})

    code = """# ── AutoML Model Training ─────────────────────────────────────────────
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              r2_score, mean_absolute_error, mean_squared_error)
import warnings
warnings.filterwarnings('ignore')

# ── Auto-detect target column ──────────────────────────────────────────────
target_keywords = ['target','label','y','class','churn','survived','outcome',
                   'response','default','fraud','converted','clicked']
target_col = None
for col in df.columns:
    if col.lower() in target_keywords:
        target_col = col
        break
if target_col is None:
    target_col = df.columns[-1]  # fallback: last column

print(f"Target column: '{target_col}'")
feature_cols = [c for c in df.columns if c != target_col]

# ── Prepare features ──────────────────────────────────────────────────────
X = df[feature_cols].copy()
y = df[target_col].copy()

# Encode categoricals in X
cat_enc = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    cat_enc[col] = le

# Fill remaining NaNs
X = X.fillna(X.median(numeric_only=True))
X = X.select_dtypes(include='number')

# Detect task type
is_classification = y.nunique() <= 20 or y.dtype in ['object', 'bool', 'category']
if y.dtype in ['object', 'category', 'bool']:
    le_y = LabelEncoder()
    y = le_y.fit_transform(y.astype(str))
    is_classification = True

print(f"Task type    : {'Classification' if is_classification else 'Regression'}")
print(f"Features     : {X.shape[1]}")
print(f"Target values: {len(np.unique(y))} unique")
print()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
    stratify=y if is_classification and len(np.unique(y)) > 1 else None)
print(f"Train/Test split: {len(X_train):,} / {len(X_test):,}")
print()

# ── Train models ──────────────────────────────────────────────────────────
results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if is_classification else 5

if is_classification:
    models_to_run = [
        ('RandomForest',      RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('GradientBoosting',  GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('LogisticRegression', Pipeline([('scaler', StandardScaler()),
                                          ('lr', LogisticRegression(max_iter=500, random_state=42))])),
    ]
    scoring = 'accuracy'

    print(f"{'Model':<25} {'Acc':>7} {'F1':>7} {'AUC':>7} {'CV±std':>12}")
    print("─" * 62)
    for name, model in models_to_run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc  = accuracy_score(y_test, preds)
        f1   = f1_score(y_test, preds, average='weighted')
        try:
            proba = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, proba, multi_class='ovr') if len(np.unique(y))>2 else roc_auc_score(y_test, proba[:,1])
        except Exception:
            auc = float('nan')
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
        print(f"{name:<25} {acc:>7.4f} {f1:>7.4f} {auc:>7.4f} {cv_scores.mean():>7.4f}±{cv_scores.std():>5.4f}")
        results.append({'model': name, 'accuracy': round(acc,4), 'f1': round(f1,4),
                         'auc': round(auc,4), 'cv_mean': round(cv_scores.mean(),4)})

else:
    models_to_run = [
        ('RandomForest',    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ('GradientBoosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('Ridge',           Pipeline([('scaler', StandardScaler()), ('r', Ridge())])),
    ]
    print(f"{'Model':<25} {'R²':>8} {'MAE':>10} {'RMSE':>10} {'CV R²':>8}")
    print("─" * 65)
    for name, model in models_to_run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2   = r2_score(y_test, preds)
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
        print(f"{name:<25} {r2:>8.4f} {mae:>10.4f} {rmse:>10.4f} {cv_scores.mean():>8.4f}")
        results.append({'model': name, 'r2': round(r2,4), 'mae': round(mae,4),
                         'rmse': round(rmse,4), 'cv_r2': round(cv_scores.mean(),4)})

print()
best = max(results, key=lambda x: x.get('accuracy', x.get('r2', 0)))
print(f"🏆 Best Model: {best['model']}")
print("✅ Training complete!")
"""

    cell = _make_cell("Modeler", code)
    _log(state, "Modeler", "Model training code generated", "success")

    return {
        **state,
        "code_cells": state.get("code_cells", []) + [cell],
        "completed_agents": state.get("completed_agents", []) + ["modeler"],
    }


def advisor_node(state: AgentState) -> AgentState:
    """Business Advisor Agent — translates findings into plain English."""
    _log(state, "Advisor", "Generating business insights...", "thinking")

    metadata = state.get("metadata", {})
    quality = metadata.get("quality", {})
    q_score = quality.get("score", 0)

    code = f"""# ── Business Advisor Agent — Plain English Insights ─────────────────
import pandas as pd
import numpy as np

print("=" * 65)
print("  BUSINESS INSIGHTS REPORT — Bro AI Advisor Agent")
print("=" * 65)
print()

num_cols = df.select_dtypes(include='number').columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()

insights = []

# ── Dataset health ─────────────────────────────────────────────────────────
quality_score = {q_score}
if quality_score >= 90:
    insights.append("✅ EXCELLENT data quality ({quality_score}/100) — this dataset is ready for analysis and modeling.")
elif quality_score >= 70:
    insights.append(f"🟡 GOOD data quality ({quality_score}/100) — minor issues may affect model accuracy.")
elif quality_score >= 50:
    insights.append(f"🟠 FAIR data quality ({quality_score}/100) — recommend cleaning before production use.")
else:
    insights.append(f"🔴 POOR data quality ({quality_score}/100) — significant cleaning required.")

# ── Scale insight ──────────────────────────────────────────────────────────
n_rows = len(df)
if n_rows < 500:
    insights.append(f"⚠  Small dataset ({{n_rows:,}} rows) — ML models may overfit. Consider collecting more data.")
elif n_rows < 10_000:
    insights.append(f"📊 Moderate dataset ({{n_rows:,}} rows) — suitable for standard ML algorithms.")
elif n_rows < 100_000:
    insights.append(f"📈 Good-sized dataset ({{n_rows:,}} rows) — can support complex models and deep learning.")
else:
    insights.append(f"🚀 Large dataset ({{n_rows:,}} rows) — consider distributed processing and sampling strategies.")

# ── Feature insights ──────────────────────────────────────────────────────
if num_cols:
    # Check for highly skewed features
    skewed = [(col, df[col].skew()) for col in num_cols if abs(df[col].skew()) > 2]
    if skewed:
        cols_str = ', '.join([f"'{c}' (skew={s:.1f})" for c,s in skewed[:3]])
        insights.append(f"📐 Highly skewed features detected: {{cols_str}}. Log-transform recommended for ML.")

    # Check for correlations
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr().abs()
        high_corr_pairs = []
        for i, col_a in enumerate(num_cols):
            for col_b in num_cols[i+1:]:
                r = corr_matrix.loc[col_a, col_b]
                if r > 0.85:
                    high_corr_pairs.append((col_a, col_b, r))
        if high_corr_pairs:
            pair_str = ', '.join([f"({{a}}, {{b}}: {{r:.2f}})" for a,b,r in high_corr_pairs[:2]])
            insights.append(f"🔗 Highly correlated feature pairs found: {{pair_str}}. Consider removing one to reduce multicollinearity.")

# ── Categorical insights ──────────────────────────────────────────────────
if cat_cols:
    imbalanced = []
    for col in cat_cols[:5]:
        vc = df[col].value_counts(normalize=True)
        if len(vc) > 1 and vc.iloc[0] > 0.8:
            imbalanced.append(col)
    if imbalanced:
        insights.append(f"⚖  Imbalanced categorical columns: {{imbalanced}}. May need stratified sampling.")

# ── Recommendations ───────────────────────────────────────────────────────
print("💡 KEY INSIGHTS:")
print()
for i, insight in enumerate(insights, 1):
    print(f"  {{i}}. {{insight}}")

print()
print("─" * 65)
print("📋 RECOMMENDATIONS:")
print()
recs = [
    "1. Run the Cleaner Agent before training any ML models.",
    "2. Use cross-validation (5-fold minimum) to prevent overfitting.",
    "3. Monitor feature importance from tree-based models for variable selection.",
    "4. Consider SHAP values for explainability in production deployments.",
    "5. Set up data drift monitoring if deploying to production.",
]
for rec in recs:
    print(f"  {{rec}}")

print()
print("=" * 65)
print("  Advisory complete. Check the Report tab for full output.")
print("=" * 65)
"""

    cell = _make_cell("Advisor", code)
    _log(state, "Advisor", "Business insights generated", "success")

    return {
        **state,
        "code_cells": state.get("code_cells", []) + [cell],
        "completed_agents": state.get("completed_agents", []) + ["advisor"],
        "finished": True,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_cell(agent: str, code: str) -> dict:
    return {
        "agent": agent,
        "code": code.strip(),
        "output": "",
        "status": "pending",
        "ts": datetime.now().isoformat(),
    }

def _log(state: AgentState, agent: str, message: str, level: str = "info") -> None:
    state.setdefault("logs", []).append({
        "ts": datetime.now().strftime("%H:%M:%S"),
        "agent": agent,
        "message": message,
        "level": level,
    })


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph():
    """
    Build the LangGraph agent graph.
    Returns a compiled graph if LangGraph is available,
    otherwise returns None (orchestrator.py handles fallback).
    """
    try:
        from langgraph.graph import StateGraph, END

        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("router",           router_node)
        graph.add_node("validator",        validator_node)
        graph.add_node("cleaner",          cleaner_node)
        graph.add_node("feature_engineer", feature_engineer_node)
        graph.add_node("analyst",          analyst_node)
        graph.add_node("modeler",          modeler_node)
        graph.add_node("advisor",          advisor_node)

        # Entry point
        graph.set_entry_point("router")

        # Router decides next step based on pipeline list
        def route_next(state: AgentState) -> str:
            pipeline = state.get("pipeline", [])
            completed = state.get("completed_agents", [])
            for agent in pipeline:
                if agent not in completed and agent in [
                    "validator","cleaner","feature_engineer","analyst","modeler","advisor"
                ]:
                    return agent
            return END

        graph.add_conditional_edges("router", route_next)
        graph.add_conditional_edges("validator", route_next)
        graph.add_conditional_edges("cleaner", route_next)
        graph.add_conditional_edges("feature_engineer", route_next)
        graph.add_conditional_edges("analyst", route_next)
        graph.add_conditional_edges("modeler", route_next)
        graph.add_edge("advisor", END)

        return graph.compile()

    except ImportError:
        return None


# Compiled graph (None if LangGraph not available)
compiled_graph = build_graph()
