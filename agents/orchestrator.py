"""
BroAI Orchestrator - LangGraph Multi-Agent System for Numdux.
Coordinates specialized agents for data science tasks.
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, TypedDict
from datetime import datetime


# ── Agent Prompts ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Bro AI, an expert data science co-pilot inside Numdux.
You are proactive, efficient, and always explain what you're doing.
You generate clean, well-commented Python code using pandas, numpy, sklearn, plotly.
Always respond with valid JSON in the exact format requested.
Never add markdown backticks around JSON. Only output the JSON object."""

VALIDATOR_PROMPT = """You are the Data Validator Agent. 
Given dataset metadata, generate Python code to:
1. Check data quality (missing values, duplicates, outliers, data types)
2. Print a quality report
3. Return a quality_score from 0-100

Respond ONLY with this JSON:
{
  "code": "# Python code here",
  "summary": "Brief description of findings",
  "issues": ["list", "of", "issues"],
  "quality_score": 85
}"""

CLEANER_PROMPT = """You are the Data Cleaner Agent.
Given dataset metadata and quality issues, generate Python code to clean the data.
The code should work on a variable called 'df' and produce 'df_clean'.
Apply best practices: handle missing values, remove duplicates, fix dtypes, handle outliers.

Respond ONLY with this JSON:
{
  "code": "# Cleaning code here",
  "summary": "What was cleaned",
  "steps": ["step1", "step2"]
}"""

ANALYST_PROMPT = """You are the Data Analyst Agent.
Perform statistical analysis on the dataset. Generate Python code for:
1. Descriptive statistics
2. Distribution analysis
3. Correlation analysis
4. Key insights using plotly for visualization

Respond ONLY with this JSON:
{
  "code": "# Analysis code",
  "insights": ["insight 1", "insight 2"],
  "key_findings": "Summary paragraph"
}"""

MODELER_PROMPT = """You are the ML Modeler Agent.
Given the dataset and task description, generate Python code to:
1. Prepare features and target
2. Train appropriate models
3. Evaluate and compare them
4. Print a model leaderboard

Respond ONLY with this JSON:
{
  "code": "# Modeling code",
  "task_type": "classification|regression",
  "models_used": ["RandomForest", "..."],
  "summary": "What was trained"
}"""

ADVISOR_PROMPT = """You are the Business Advisor Agent.
Given dataset metadata and analysis results, provide:
1. Plain English business insights
2. Actionable recommendations
3. Potential risks or concerns

Respond ONLY with this JSON:
{
  "insights": ["insight 1", "insight 2"],
  "recommendations": ["rec 1", "rec 2"],
  "risks": ["risk 1"],
  "executive_summary": "2-3 sentence summary"
}"""

TASK_ROUTER_PROMPT = """You are the Task Router for Bro AI.
Given a user task and dataset metadata, decide which agents to run and in what order.
Available agents: validator, cleaner, analyst, feature_engineer, modeler, advisor

Respond ONLY with this JSON:
{
  "pipeline": ["validator", "cleaner", "analyst"],
  "task_type": "eda|classification|regression|clustering|insights",
  "target_column": "column_name_or_null",
  "reasoning": "Why this pipeline"
}"""


class BroAIOrchestrator:
    """
    LangGraph-based multi-agent orchestrator for Bro AI.
    Falls back to rule-based mode if LLM is unavailable.
    """

    def __init__(self, provider: str = "ollama", model: str = "llama3.2",
                 api_key: str = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self._llm = None

    def _get_llm(self):
        """Lazy-initialize LLM via LiteLLM."""
        if self._llm is not None:
            return self._llm
        try:
            from litellm import completion
            self._llm = completion
            return self._llm
        except ImportError:
            return None

    def _call_llm(self, system: str, user: str) -> Optional[str]:
        """Make a single LLM call, return text response."""
        llm = self._get_llm()
        if llm is None:
            return None

        # Build model string for LiteLLM
        if self.provider == "ollama":
            model_str = f"ollama/{self.model}"
        elif self.provider == "openai":
            model_str = self.model
        elif self.provider == "anthropic":
            model_str = self.model
        elif self.provider == "groq":
            model_str = f"groq/{self.model}"
        else:
            model_str = self.model

        try:
            response = llm(
                model=model_str,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
                max_tokens=2000,
                api_key=self.api_key,
            )
            return response.choices[0].message.content
        except Exception as e:
            return None

    def _parse_json_response(self, text: str) -> Optional[dict]:
        """Robustly parse JSON from LLM response."""
        if not text:
            return None
        try:
            # Strip markdown code blocks if present
            text = text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in text
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
        return None

    def run(self, df, metadata: dict, task: str = "full_analysis") -> dict:
        """
        Main entry point. Run the agent pipeline on a dataset.
        
        Returns dict with code_cells and logs.
        """
        code_cells = []
        logs = []

        def log(agent, msg, level="info"):
            logs.append({"agent": agent, "message": msg, "level": level,
                         "ts": datetime.now().strftime("%H:%M:%S")})

        log("Orchestrator", f"Starting pipeline for task: '{task}'", "info")

        # 1. Route the task
        pipeline = self._route_task(task, metadata, log)

        # 2. Run each agent
        for agent_name in pipeline:
            log(agent_name.title(), f"Starting {agent_name} agent...", "thinking")
            cell = self._run_agent(agent_name, df, metadata, task, log)
            if cell:
                code_cells.append(cell)
                log(agent_name.title(), f"Generated code cell", "success")

        log("Orchestrator", f"Pipeline complete. Generated {len(code_cells)} cells.", "success")

        return {"code_cells": code_cells, "logs": logs}

    def _route_task(self, task: str, metadata: dict, log) -> List[str]:
        """Decide which agents to run based on task."""
        # Try LLM routing first
        user_msg = f"Task: '{task}'\nDataset: {json.dumps({k: metadata.get(k) for k in ['filename','rows','cols','dtypes_summary']}, indent=2)}"
        raw = self._call_llm(TASK_ROUTER_PROMPT + "\n" + SYSTEM_PROMPT, user_msg)
        parsed = self._parse_json_response(raw) if raw else None

        if parsed and "pipeline" in parsed:
            log("Router", f"LLM selected pipeline: {parsed['pipeline']}", "info")
            return parsed["pipeline"]

        # Rule-based fallback
        task_lower = task.lower()
        if any(w in task_lower for w in ["clean", "quality", "fix"]):
            return ["validator", "cleaner"]
        elif any(w in task_lower for w in ["predict", "model", "classify", "regress", "churn", "train"]):
            return ["validator", "cleaner", "analyst", "modeler", "advisor"]
        elif any(w in task_lower for w in ["insight", "business", "summary", "analyze"]):
            return ["analyst", "advisor"]
        else:
            return ["validator", "cleaner", "analyst"]

    def _run_agent(self, agent_name: str, df, metadata: dict, task: str, log) -> Optional[dict]:
        """Run a single agent, return a code cell dict."""
        prompts = {
            "validator": (VALIDATOR_PROMPT, self._validator_fallback),
            "cleaner": (CLEANER_PROMPT, self._cleaner_fallback),
            "analyst": (ANALYST_PROMPT, self._analyst_fallback),
            "modeler": (MODELER_PROMPT, self._modeler_fallback),
            "advisor": (ADVISOR_PROMPT, None),
        }

        if agent_name not in prompts:
            return None

        system_prompt, fallback_fn = prompts[agent_name]
        meta_summary = json.dumps({
            "filename": metadata.get("filename"),
            "rows": metadata.get("rows"),
            "cols": metadata.get("cols"),
            "columns": {k: v.get("dtype") for k, v in metadata.get("columns", {}).items()},
            "quality": metadata.get("quality"),
        }, indent=2)

        user_msg = f"Task: '{task}'\nDataset metadata:\n{meta_summary}"
        raw = self._call_llm(SYSTEM_PROMPT + "\n\n" + system_prompt, user_msg)
        parsed = self._parse_json_response(raw) if raw else None

        if parsed and "code" in parsed:
            return {
                "agent": agent_name.title(),
                "code": parsed["code"],
                "output": "",
                "status": "pending",
                "metadata": {k: v for k, v in parsed.items() if k != "code"},
            }
        elif fallback_fn:
            log(agent_name.title(), "LLM unavailable, using rule-based fallback", "warning")
            return fallback_fn(metadata)
        return None

    # ── Rule-based fallbacks ──────────────────────────────────────────────────

    def _validator_fallback(self, metadata: dict) -> dict:
        return {
            "agent": "Validator",
            "code": """# ── Data Quality Validation ─────────────────────────────
import pandas as pd

print("=" * 60)
print("DATA QUALITY REPORT")
print("=" * 60)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Memory: {df.memory_usage(deep=True).sum()/1e6:.2f} MB")
print()

# Missing values
missing = df.isnull().sum()
missing_pct = (missing/len(df)*100).round(2)
has_missing = missing[missing>0]
print(f"Missing values: {missing.sum():,} total ({missing.sum()/df.size*100:.1f}% of all cells)")
if len(has_missing) > 0:
    print(has_missing.to_string())
else:
    print("  No missing values! ✅")

print()
# Duplicates
dups = df.duplicated().sum()
print(f"Duplicate rows: {dups:,} ({dups/len(df)*100:.1f}%)")

# Data types
print()
print("Column dtypes:")
print(df.dtypes.value_counts())

# Numeric outliers (IQR method)
print()
num_cols = df.select_dtypes(include='number').columns
outlier_counts = {}
for col in num_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = ((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).sum()
    if outliers > 0:
        outlier_counts[col] = outliers

if outlier_counts:
    print("Potential outliers (IQR method):")
    for col, n in outlier_counts.items():
        print(f"  {col}: {n} ({n/len(df)*100:.1f}%)")
else:
    print("No significant outliers detected. ✅")
""",
            "output": "", "status": "pending",
        }

    def _cleaner_fallback(self, metadata: dict) -> dict:
        return {
            "agent": "Cleaner",
            "code": """# ── Auto Data Cleaning Pipeline ─────────────────────────
import pandas as pd
import numpy as np

df_clean = df.copy()
original_shape = df_clean.shape
log = []

# 1. Remove fully empty rows/columns
df_clean.dropna(how='all', inplace=True)
df_clean.dropna(axis=1, how='all', inplace=True)

# 2. Remove duplicates
n_dups = df_clean.duplicated().sum()
df_clean.drop_duplicates(inplace=True)
log.append(f"Removed {n_dups} duplicate rows")

# 3. Fix obvious dtype issues
for col in df_clean.columns:
    if df_clean[col].dtype == object:
        # Try to convert to numeric
        converted = pd.to_numeric(df_clean[col], errors='coerce')
        if converted.notna().sum() / len(df_clean) > 0.8:
            df_clean[col] = converted
            log.append(f"Converted '{col}' to numeric")

# 4. Fill missing values
num_cols = df_clean.select_dtypes(include='number').columns
for col in num_cols:
    n_missing = df_clean[col].isnull().sum()
    if n_missing > 0:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
        log.append(f"Filled {n_missing} missing values in '{col}' with median")

cat_cols = df_clean.select_dtypes(include='object').columns
for col in cat_cols:
    n_missing = df_clean[col].isnull().sum()
    if n_missing > 0 and len(df_clean[col].mode()) > 0:
        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        log.append(f"Filled {n_missing} missing in '{col}' with mode")

# 5. Strip whitespace from string columns
for col in df_clean.select_dtypes(include='object').columns:
    df_clean[col] = df_clean[col].str.strip()

print("=" * 50)
print("CLEANING SUMMARY")
print("=" * 50)
print(f"Original shape: {original_shape}")
print(f"Clean shape:    {df_clean.shape}")
print(f"Rows removed:   {original_shape[0] - df_clean.shape[0]}")
print()
print("Steps applied:")
for step in log:
    print(f"  ✓ {step}")

print()
print(f"Remaining nulls: {df_clean.isnull().sum().sum()}")
df = df_clean  # Update working dataframe
""",
            "output": "", "status": "pending",
        }

    def _analyst_fallback(self, metadata: dict) -> dict:
        return {
            "agent": "Analyst",
            "code": """# ── Exploratory Data Analysis ───────────────────────────
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

num_cols = df.select_dtypes(include='number').columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)
print(f"Numeric columns  ({len(num_cols)}): {num_cols}")
print(f"Categorical cols ({len(cat_cols)}): {cat_cols}")
print()

if num_cols:
    print("── Numeric Summary ──────────────────────────────────────")
    print(df[num_cols].describe().round(4).to_string())
    print()

    if len(num_cols) >= 2:
        print("── Top Correlations ────────────────────────────────────")
        corr = df[num_cols].corr()
        pairs = (corr.where(~(corr==1.0)).unstack().dropna()
                     .reset_index().rename(columns={'level_0':'A','level_1':'B',0:'r'})
                     .assign(abs_r=lambda x: x['r'].abs())
                     .sort_values('abs_r', ascending=False))
        # Remove duplicate pairs
        seen = set()
        unique_pairs = []
        for _, row in pairs.iterrows():
            key = frozenset([row['A'], row['B']])
            if key not in seen:
                seen.add(key)
                unique_pairs.append(row)
        top = pd.DataFrame(unique_pairs[:5])
        print(top[['A','B','r']].to_string(index=False))
        print()

if cat_cols:
    print("── Categorical Summary ─────────────────────────────────")
    for col in cat_cols[:5]:
        vc = df[col].value_counts()
        print(f"  {col}: {df[col].nunique()} unique values")
        print(f"    Top 3: {dict(vc.head(3))}")
    print()

# Quick insight generation
print("── Key Insights ────────────────────────────────────────")
insights = []
if df.isnull().sum().sum() == 0:
    insights.append("✅ Dataset is complete — no missing values.")
else:
    worst_col = df.isnull().mean().idxmax()
    insights.append(f"⚠ '{worst_col}' has the most missing data ({df[worst_col].isnull().mean()*100:.1f}%).")

if num_cols:
    for col in num_cols[:3]:
        skew = df[col].skew()
        if abs(skew) > 1:
            insights.append(f"⚠ '{col}' is {'right' if skew>0 else 'left'}-skewed (skewness={skew:.2f}).")

for insight in insights:
    print(f"  {insight}")
""",
            "output": "", "status": "pending",
        }

    def _modeler_fallback(self, metadata: dict) -> dict:
        return {
            "agent": "Modeler",
            "code": """# ── AutoML Model Training ───────────────────────────────
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ── Auto-detect target column ───────────────────────────
# Heuristic: last column or column with 'target','label','y' in name
target_col = None
for col in df.columns:
    if col.lower() in ['target','label','y','class','churn','survived']:
        target_col = col
        break
if target_col is None:
    target_col = df.columns[-1]

print(f"Target column: '{target_col}'")
feature_cols = [c for c in df.columns if c != target_col]

X = df[feature_cols].copy()
y = df[target_col].copy()

# Encode categoricals
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
X = X.fillna(X.median(numeric_only=True))

is_classification = y.nunique() <= 20 or y.dtype == object
if y.dtype == object:
    y = LabelEncoder().fit_transform(y.astype(str))
    is_classification = True

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Task: {'Classification' if is_classification else 'Regression'}")
print()

results = []
if is_classification:
    models = [
        ('RandomForest', RandomForestClassifier(100, random_state=42)),
        ('GradientBoosting', GradientBoostingClassifier(100, random_state=42)),
        ('LogisticRegression', LogisticRegression(max_iter=500)),
    ]
    print(f"{'Model':<25} {'Accuracy':>10} {'CV Score':>10}")
    print('-' * 48)
    for name, model in models:
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        cv = cross_val_score(model, X, y, cv=3).mean()
        print(f"{name:<25} {acc:>10.4f} {cv:>10.4f}")
        results.append({'model': name, 'accuracy': round(acc, 4), 'cv_score': round(cv, 4)})
else:
    models = [
        ('RandomForest', RandomForestRegressor(100, random_state=42)),
        ('Ridge', Ridge()),
    ]
    print(f"{'Model':<25} {'R²':>10} {'MAE':>12}")
    print('-' * 50)
    for name, model in models:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds); mae = mean_absolute_error(y_test, preds)
        print(f"{name:<25} {r2:>10.4f} {mae:>12.4f}")
        results.append({'model': name, 'r2': round(r2, 4), 'mae': round(mae, 4)})

print()
print("✅ Training complete!")
""",
            "output": "", "status": "pending",
        }
