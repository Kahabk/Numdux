"""
Numdux - Intelligent AI Co-Pilot for Data Scientists
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="Numdux · AI Data Co-Pilot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@300;400;500;700&display=swap');
:root {
    --bg-primary:#0a0e1a; --bg-secondary:#111827; --bg-card:#1a2035;
    --accent-cyan:#00d4ff; --accent-purple:#7c3aed; --accent-green:#10b981;
    --accent-orange:#f59e0b; --accent-red:#ef4444;
    --text-primary:#e2e8f0; --text-secondary:#94a3b8; --border:#1e2d45;
    --font-mono:'JetBrains Mono',monospace; --font-sans:'DM Sans',sans-serif;
}
html,body,[class*="css"]{font-family:var(--font-sans)!important;}
.main .block-container{padding-top:1rem;max-width:100%;}
.stApp{background:var(--bg-primary);}
section[data-testid="stSidebar"]{background:var(--bg-secondary)!important;border-right:1px solid var(--border)!important;}
.numdux-header{display:flex;align-items:center;gap:12px;padding:0.5rem 0 1.5rem 0;border-bottom:1px solid var(--border);margin-bottom:1.5rem;}
.numdux-logo{font-family:var(--font-mono);font-size:1.8rem;font-weight:600;background:linear-gradient(135deg,var(--accent-cyan),var(--accent-purple));-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-1px;}
.numdux-tagline{font-size:0.75rem;color:var(--text-secondary);text-transform:uppercase;letter-spacing:2px;}
.stat-card{background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:1rem 1.25rem;margin-bottom:0.75rem;}
.stat-card h3{color:var(--text-secondary);font-size:0.7rem;text-transform:uppercase;letter-spacing:1.5px;margin:0 0 0.25rem 0;}
.stat-card .value{color:var(--text-primary);font-size:1.5rem;font-family:var(--font-mono);font-weight:600;}
.agent-msg{background:var(--bg-card);border-left:3px solid var(--accent-cyan);border-radius:0 8px 8px 0;padding:0.75rem 1rem;margin:0.5rem 0;font-size:0.875rem;color:var(--text-primary);}
.agent-msg.thinking{border-left-color:var(--accent-purple);opacity:0.8;}
.agent-msg.success{border-left-color:var(--accent-green);}
.agent-msg.warning{border-left-color:var(--accent-orange);}
.agent-msg.error{border-left-color:var(--accent-red);}
.agent-name{font-size:0.65rem;text-transform:uppercase;letter-spacing:1.5px;color:var(--accent-cyan);font-family:var(--font-mono);margin-bottom:0.25rem;}
.code-cell{background:#0d1117;border:1px solid var(--border);border-radius:8px;margin:0.75rem 0;overflow:hidden;}
.code-cell-header{display:flex;justify-content:space-between;align-items:center;padding:0.4rem 0.75rem;background:var(--bg-card);border-bottom:1px solid var(--border);}
.cell-badge{font-family:var(--font-mono);font-size:0.65rem;color:var(--text-secondary);}
.upload-zone{border:2px dashed var(--border);border-radius:16px;padding:3rem 2rem;text-align:center;background:var(--bg-card);margin:1rem 0;}
.upload-zone:hover{border-color:var(--accent-cyan);}
.upload-icon{font-size:3rem;margin-bottom:0.5rem;}
.upload-title{font-size:1.1rem;font-weight:500;color:var(--text-primary);margin-bottom:0.25rem;}
.upload-subtitle{color:var(--text-secondary);font-size:0.85rem;}
.progress-step{display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0;font-size:0.85rem;}
.step-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.step-dot.done{background:var(--accent-green);}
.step-dot.active{background:var(--accent-cyan);animation:pulse 1s infinite;}
.step-dot.pending{background:var(--border);}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.3;}}
.stTabs [data-baseweb="tab-list"]{background:var(--bg-secondary)!important;border-bottom:1px solid var(--border)!important;gap:0!important;}
.stTabs [data-baseweb="tab"]{color:var(--text-secondary)!important;font-size:0.8rem!important;padding:0.6rem 1.2rem!important;}
.stTabs [aria-selected="true"]{color:var(--accent-cyan)!important;border-bottom:2px solid var(--accent-cyan)!important;}
.stButton>button{background:var(--bg-card)!important;color:var(--text-primary)!important;border:1px solid var(--border)!important;border-radius:8px!important;font-family:var(--font-sans)!important;font-size:0.8rem!important;transition:all 0.15s!important;}
.stButton>button:hover{border-color:var(--accent-cyan)!important;color:var(--accent-cyan)!important;}
[data-testid="metric-container"]{background:var(--bg-card)!important;border:1px solid var(--border)!important;border-radius:10px!important;padding:0.75rem!important;}
[data-testid="stMetricValue"]{color:var(--text-primary)!important;}
[data-testid="stMetricLabel"]{color:var(--text-secondary)!important;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg-primary);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:var(--accent-cyan);}
h1,h2,h3,h4,h5,h6{color:var(--text-primary)!important;}
p,li,span{color:var(--text-secondary);}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "dataset": None, "df": None, "metadata": None,
        "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "agent_log": [], "code_cells": [], "reports": [], "models": [],
        "bro_ai_running": False, "bro_ai_paused": False,
        "pipeline_steps": [], "eda_results": None, "chat_history": [],
        "llm_provider": "ollama", "llm_model": "llama3.2", "auto_run": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ── Utilities ─────────────────────────────────────────────────────────────────
def add_agent_log(agent, message, level="info"):
    st.session_state.agent_log.append({
        "ts": datetime.now().strftime("%H:%M:%S"),
        "agent": agent, "message": message, "level": level,
    })

def add_code_cell(code, output="", agent="BroAI", status="pending"):
    st.session_state.code_cells.append({
        "id": len(st.session_state.code_cells),
        "agent": agent, "code": code, "output": output,
        "status": status, "ts": datetime.now().isoformat(),
    })

def compute_quality_score(df):
    total_cells = df.shape[0] * df.shape[1]
    missing = df.isnull().sum().sum()
    missing_pct = (missing / total_cells * 100) if total_cells > 0 else 0
    dup_rows = df.duplicated().sum()
    dup_pct = (dup_rows / len(df) * 100) if len(df) > 0 else 0
    score = 100 - min(missing_pct * 1.5, 40) - min(dup_pct * 0.5, 15)
    mixed = sum(1 for col in df.columns if df[col].dtype == object)
    score -= min(mixed * 0.5, 10)
    return {
        "score": max(0, round(score, 1)),
        "missing_pct": round(missing_pct, 2),
        "missing_count": int(missing),
        "duplicate_rows": int(dup_rows),
        "dup_pct": round(dup_pct, 2),
        "total_cells": total_cells,
        "shape": df.shape,
    }

def extract_metadata(df, filename):
    quality = compute_quality_score(df)
    col_info = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        col_info[col] = {
            "dtype": dtype,
            "null_count": int(df[col].isnull().sum()),
            "null_pct": round(df[col].isnull().mean() * 100, 2),
            "unique_count": int(df[col].nunique()),
            "sample": str(df[col].dropna().iloc[0]) if df[col].dropna().shape[0] > 0 else None,
        }
        if dtype in ["int64", "float64", "int32", "float32"]:
            col_info[col].update({
                "min": float(df[col].min()) if not df[col].isnull().all() else None,
                "max": float(df[col].max()) if not df[col].isnull().all() else None,
                "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
            })
    return {
        "filename": filename, "rows": df.shape[0], "cols": df.shape[1],
        "columns": col_info, "quality": quality,
        "dtypes_summary": df.dtypes.astype(str).value_counts().to_dict(),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
        "loaded_at": datetime.now().isoformat(),
    }

def load_dataframe(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        content = uploaded_file.read(4096).decode("utf-8", errors="replace")
        uploaded_file.seek(0)
        sep = "," if content.count(",") >= content.count(";") else ";"
        return pd.read_csv(uploaded_file, sep=sep, low_memory=False)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    elif name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    elif name.endswith(".json"):
        return pd.read_json(uploaded_file)
    elif name.endswith(".tsv"):
        return pd.read_csv(uploaded_file, sep="\t")
    return pd.read_csv(uploaded_file)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:0.5rem 0 1rem 0;border-bottom:1px solid var(--border);">
            <div style="font-family:var(--font-mono);font-size:1.3rem;font-weight:600;
                        background:linear-gradient(135deg,#00d4ff,#7c3aed);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                🧠 Numdux
            </div>
            <div style="font-size:0.65rem;color:var(--text-secondary);letter-spacing:2px;
                        text-transform:uppercase;margin-top:2px;">AI Data Co-Pilot</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### ⚙️ LLM Configuration")
        provider = st.selectbox("Provider", ["ollama","openai","anthropic","groq","mistral"], key="sb_provider")
        st.session_state.llm_provider = provider

        if provider == "ollama":
            st.session_state.llm_model = st.text_input("Model", value=st.session_state.llm_model, key="sb_model")
            st.caption("Ensure Ollama is running: `ollama serve`")
        elif provider == "openai":
            st.session_state.llm_model = st.selectbox("Model", ["gpt-4o","gpt-4o-mini"], key="sb_model_oai")
            st.text_input("API Key", type="password", key="openai_key")
        elif provider == "anthropic":
            st.session_state.llm_model = st.selectbox("Model", ["claude-sonnet-4-5","claude-haiku-4-5"], key="sb_model_ant")
            st.text_input("API Key", type="password", key="anthropic_key")
        elif provider == "groq":
            st.session_state.llm_model = st.selectbox("Model", ["llama-3.1-70b-versatile","mixtral-8x7b-32768"], key="sb_model_groq")
            st.text_input("API Key", type="password", key="groq_key")
        else:
            st.session_state.llm_model = st.text_input("Model Name", value="mistral-7b", key="sb_model_custom")

        st.divider()
        st.markdown("### 🤖 Bro AI Controls")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.auto_run = st.toggle("Auto-Run", value=st.session_state.auto_run)
        with col2:
            if st.session_state.bro_ai_running:
                if st.button("⏸ Pause"): st.session_state.bro_ai_paused = True
            else:
                if st.button("▶ Run"):
                    st.session_state.bro_ai_running = True
                    st.session_state.bro_ai_paused = False

        if st.session_state.df is not None:
            st.markdown("### 💬 Chat with Bro AI")
            user_prompt = st.text_area(
                "Task", placeholder='"Build churn model", "Clean for ML", "Find insights"',
                height=80, key="chat_prompt_input"
            )
            if st.button("Send 🚀", key="send_chat") and user_prompt.strip():
                st.session_state.chat_history.append({"role": "user", "content": user_prompt})
                add_agent_log("BroAI", f"Received: {user_prompt}", "info")
                # Dispatch to orchestrator
                try:
                    from agents.orchestrator import BroAIOrchestrator
                    orch = BroAIOrchestrator(
                        provider=st.session_state.llm_provider,
                        model=st.session_state.llm_model,
                    )
                    with st.spinner("🤖 Bro AI thinking..."):
                        result = orch.run(
                            df=st.session_state.df,
                            metadata=st.session_state.metadata,
                            task=user_prompt,
                        )
                    for cell in result.get("code_cells", []):
                        add_code_cell(cell["code"], cell.get("output",""), cell.get("agent","BroAI"), cell.get("status","pending"))
                    for log in result.get("logs", []):
                        add_agent_log(log["agent"], log["message"], log.get("level","info"))
                    st.rerun()
                except Exception as e:
                    add_agent_log("BroAI", f"LLM not available: {e}", "warning")

        st.divider()
        if st.session_state.df is not None:
            df = st.session_state.df
            q = (st.session_state.metadata or {}).get("quality", {})
            qcolor = '#10b981' if q.get('score',0)>75 else '#f59e0b' if q.get('score',0)>50 else '#ef4444'
            st.markdown(f"""
            <div class="stat-card"><h3>Dataset</h3>
                <div class="value">{df.shape[0]:,} × {df.shape[1]}</div></div>
            <div class="stat-card"><h3>Quality Score</h3>
                <div class="value" style="color:{qcolor}">{q.get('score','—')}</div></div>
            """, unsafe_allow_html=True)
            if st.button("🗑 Clear Session"):
                for key in ["dataset","df","metadata","agent_log","code_cells","reports","models","pipeline_steps","eda_results"]:
                    st.session_state[key] = [] if key in ["agent_log","code_cells","reports","models","pipeline_steps"] else None
                st.rerun()

        if st.session_state.agent_log:
            st.divider()
            st.markdown("### 📋 Activity")
            for entry in reversed(st.session_state.agent_log[-6:]):
                lc = {"info":"#00d4ff","thinking":"#7c3aed","success":"#10b981","warning":"#f59e0b","error":"#ef4444"}.get(entry["level"],"#94a3b8")
                msg = entry['message'][:55] + "…" if len(entry['message']) > 55 else entry['message']
                st.markdown(f"<div style='font-size:0.72rem;padding:2px 0;'><span style='color:{lc};font-family:var(--font-mono);'>[{entry['ts']}]</span> <span style='color:#e2e8f0;'>{msg}</span></div>", unsafe_allow_html=True)


# ── Upload Tab ────────────────────────────────────────────────────────────────
def render_upload_tab():
    st.markdown("<h2 style='font-size:1.5rem;'>📂 Upload Dataset</h2>", unsafe_allow_html=True)
    col_upload, col_info = st.columns([3, 2], gap="large")
    with col_upload:
        uploaded = st.file_uploader(
            "Upload Dataset", type=["csv","xlsx","xls","parquet","json","tsv"],
            label_visibility="collapsed", key="file_uploader"
        )
        st.markdown("""
        <div class="upload-zone">
            <div class="upload-icon">📊</div>
            <div class="upload-title">Drop your dataset here</div>
            <div class="upload-subtitle">CSV · Excel · Parquet · JSON · TSV</div>
        </div>""", unsafe_allow_html=True)

        if uploaded and (st.session_state.dataset is None or st.session_state.dataset != uploaded.name):
            with st.spinner("🔍 Reading dataset..."):
                try:
                    df = load_dataframe(uploaded)
                    metadata = extract_metadata(df, uploaded.name)
                    st.session_state.df = df
                    st.session_state.dataset = uploaded.name
                    st.session_state.metadata = metadata
                    add_agent_log("DataLoader", f"Loaded {uploaded.name} → {df.shape[0]:,}×{df.shape[1]}", "success")
                    add_agent_log("Validator", f"Quality={metadata['quality']['score']} | Missing={metadata['quality']['missing_pct']}% | Dups={metadata['quality']['dup_pct']}%", "info")
                    if st.session_state.auto_run:
                        st.session_state.bro_ai_running = True
                        add_agent_log("BroAI", "Auto-starting analysis pipeline 🚀", "info")
                        # Generate initial EDA code cells
                        _generate_initial_cells(df, uploaded.name)
                    st.success(f"✅ Loaded **{uploaded.name}** — {df.shape[0]:,} rows, {df.shape[1]} cols")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {e}")

    with col_info:
        st.markdown("**Supported Formats**")
        for fmt, desc in [("CSV / TSV","Comma or tab-separated"),("Excel",".xlsx and .xls"),("Parquet","Columnar binary"),("JSON","Flat or record-oriented")]:
            st.markdown(f"<div style='padding:6px 0;border-bottom:1px solid var(--border);'><span style='color:var(--accent-cyan);font-family:var(--font-mono);font-size:0.8rem;'>{fmt}</span><span style='color:var(--text-secondary);font-size:0.75rem;margin-left:8px;'>{desc}</span></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Bro AI will automatically:**")
        for s in ["✦ Validate data quality","✦ Generate full EDA","✦ Detect & clean issues","✦ Engineer features","✦ Train best models","✦ Generate insights","✦ Build full report"]:
            st.markdown(f"<div style='color:var(--text-secondary);font-size:0.8rem;padding:2px 0;'>{s}</div>", unsafe_allow_html=True)

    if st.session_state.df is not None:
        df = st.session_state.df
        q = st.session_state.metadata["quality"]
        st.divider()
        st.markdown("### 📋 Dataset Preview")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Quality Score", f"{q['score']}/100")
        c2.metric("Rows", f"{q['shape'][0]:,}")
        c3.metric("Columns", f"{q['shape'][1]}")
        c4.metric("Missing %", f"{q['missing_pct']}%")
        c5.metric("Duplicates", f"{q['duplicate_rows']:,}")
        with st.expander("📄 Raw Data (first 100 rows)", expanded=True):
            st.dataframe(df.head(100), use_container_width=True, height=300)
        with st.expander("🔬 Column Inspector"):
            cols_df = pd.DataFrame([{
                "Column": col, "Type": str(df[col].dtype),
                "Non-Null": int(df[col].notna().sum()),
                "Null %": f"{df[col].isnull().mean()*100:.1f}%",
                "Unique": df[col].nunique(),
                "Sample": str(df[col].dropna().iloc[0]) if df[col].dropna().shape[0]>0 else "—",
            } for col in df.columns])
            st.dataframe(cols_df, use_container_width=True, hide_index=True)

def _generate_initial_cells(df, filename):
    """Auto-generate starter code cells when dataset is uploaded."""
    cells = [
        {
            "agent": "Validator",
            "code": f"""# ── Data Quality Report ─────────────────────────────────
import pandas as pd

print(f"Shape: {{df.shape}}")
print(f"\\nDtype breakdown:")
print(df.dtypes.value_counts())
print(f"\\nMissing values:")
missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False).head(20))
print(f"\\nDuplicate rows: {{df.duplicated().sum()}}")
print(f"\\nMemory usage: {{df.memory_usage(deep=True).sum()/1e6:.2f}} MB")
"""
        },
        {
            "agent": "Cleaner",
            "code": f"""# ── Auto Cleaning Pipeline ───────────────────────────────
import pandas as pd

df_clean = df.copy()

# 1. Drop fully empty rows/cols
df_clean = df_clean.dropna(how='all', axis=0)
df_clean = df_clean.dropna(how='all', axis=1)

# 2. Remove exact duplicates
n_dups = df_clean.duplicated().sum()
df_clean = df_clean.drop_duplicates()
print(f"Removed {{n_dups}} duplicate rows")

# 3. Fill numeric NaNs with median
num_cols = df_clean.select_dtypes(include='number').columns
for col in num_cols:
    if df_clean[col].isnull().any():
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# 4. Fill categorical NaNs with mode
cat_cols = df_clean.select_dtypes(include='object').columns
for col in cat_cols:
    if df_clean[col].isnull().any():
        mode_val = df_clean[col].mode()
        if len(mode_val) > 0:
            df_clean[col] = df_clean[col].fillna(mode_val[0])

print(f"Clean shape: {{df_clean.shape}}")
print(f"Remaining nulls: {{df_clean.isnull().sum().sum()}}")
df = df_clean  # Update working dataframe
"""
        },
        {
            "agent": "Analyst",
            "code": f"""# ── Statistical Summary ──────────────────────────────────
import pandas as pd
import plotly.express as px

num_cols = df.select_dtypes(include='number').columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()

print(f"Numeric columns ({{len(num_cols)}}): {{num_cols}}")
print(f"Categorical columns ({{len(cat_cols)}}): {{cat_cols}}")
print()
print(df[num_cols].describe().round(4))

# Correlation matrix for numeric cols
if len(num_cols) >= 2:
    corr = df[num_cols].corr()
    top_pairs = (corr.where(~(corr==1.0)).unstack().dropna()
                     .reset_index().rename(columns={{'level_0':'A','level_1':'B',0:'r'}})
                     .assign(abs_r=lambda x: x['r'].abs())
                     .sort_values('abs_r', ascending=False).head(5))
    print("\\nTop 5 correlations:")
    print(top_pairs[['A','B','r']].to_string(index=False))
"""
        },
    ]
    for cell in cells:
        add_code_cell(cell["code"], agent=cell["agent"], status="pending")
    add_agent_log("BroAI", f"Generated {len(cells)} starter code cells", "success")


# ── Sandbox Tab ───────────────────────────────────────────────────────────────
def render_sandbox_tab():
    st.markdown("<h2 style='font-size:1.5rem;'>🔬 Code Sandbox</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:var(--text-secondary);font-size:0.875rem;'>Jupyter-style cells generated by Bro AI. Review, edit, and run.</p>", unsafe_allow_html=True)

    if st.session_state.df is None:
        st.info("👆 Upload a dataset first.")
        return

    with st.expander("➕ Add Custom Cell"):
        custom_code = st.text_area("Python code", value="# Your code here\nprint(df.shape)", height=120, key="custom_cell_code")
        if st.button("▶ Run Cell", key="run_custom"):
            add_code_cell(custom_code, agent="User", status="running")
            from tools.sandbox import SafeSandbox
            result = SafeSandbox().run(custom_code, {"df": st.session_state.df})
            cell = st.session_state.code_cells[-1]
            cell["output"] = result["output"]
            cell["status"] = "success" if result["success"] else "error"
            st.rerun()

    if not st.session_state.code_cells:
        st.markdown("<div style='background:var(--bg-card);border:1px dashed var(--border);border-radius:12px;padding:2rem;text-align:center;color:var(--text-secondary);'>No code cells yet. Upload a dataset and Bro AI will populate them automatically.</div>", unsafe_allow_html=True)
        return

    # Run All button
    col_a, col_b = st.columns([1, 6])
    with col_a:
        if st.button("▶▶ Run All", key="run_all"):
            from tools.sandbox import SafeSandbox
            sandbox = SafeSandbox()
            ctx = {"df": st.session_state.df}
            for cell in st.session_state.code_cells:
                if cell["status"] in ["pending", "error"]:
                    result = sandbox.run(cell["code"], ctx)
                    cell["output"] = result["output"]
                    cell["status"] = "success" if result["success"] else "error"
                    if result["success"] and "df" in result.get("locals", {}):
                        ctx["df"] = result["locals"]["df"]
            st.rerun()

    for i, cell in enumerate(st.session_state.code_cells):
        sc = {"pending":"#94a3b8","running":"#f59e0b","success":"#10b981","error":"#ef4444","approved":"#7c3aed"}.get(cell["status"],"#94a3b8")
        st.markdown(f"""<div class="code-cell"><div class="code-cell-header">
            <span class="cell-badge">In [{i+1}] · {cell['agent']}</span>
            <span style="color:{sc};font-size:0.7rem;font-family:var(--font-mono);">● {cell['status'].upper()}</span>
        </div></div>""", unsafe_allow_html=True)

        edited = st.text_area(f"cell_{i}", value=cell["code"],
            height=max(80, cell["code"].count("\n")*20+60),
            key=f"cell_code_{i}", label_visibility="collapsed")
        st.session_state.code_cells[i]["code"] = edited

        b1,b2,b3,b4,_ = st.columns([1,1,1,1,5])
        with b1:
            if st.button("▶", key=f"run_{i}", help="Run"):
                from tools.sandbox import SafeSandbox
                r = SafeSandbox().run(edited, {"df": st.session_state.df})
                cell["output"] = r["output"]; cell["status"] = "success" if r["success"] else "error"
                if r["success"] and "df" in r.get("locals",{}): st.session_state.df = r["locals"]["df"]
                st.rerun()
        with b2:
            if st.button("✓", key=f"ok_{i}", help="Approve"): cell["status"]="approved"; st.rerun()
        with b3:
            if st.button("🗑", key=f"del_{i}", help="Delete"):
                st.session_state.code_cells.pop(i); st.rerun()

        if cell.get("output"):
            with st.expander(f"Output [{i+1}]", expanded=cell["status"]=="error"):
                if cell["status"] == "error": st.error(cell["output"])
                else: st.code(cell["output"], language="text")

# ── EDA Tab ───────────────────────────────────────────────────────────────────
def render_eda_tab():
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.markdown("<h2 style='font-size:1.5rem;'>📈 Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    if st.session_state.df is None:
        st.info("👆 Upload a dataset first."); return

    df = st.session_state.df
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()

    t1,t2,t3,t4 = st.tabs(["📊 Distributions","🔗 Correlations","🕳 Missing","📐 Stats"])

    with t1:
        if num_cols:
            col_select = st.selectbox("Select numeric column", num_cols, key="dist_col")
            col = df[col_select].dropna()
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Histogram","Box Plot"])
            fig.add_trace(go.Histogram(x=col, name="Hist", marker_color="#00d4ff", opacity=0.75), row=1, col=1)
            fig.add_trace(go.Box(y=col, name="Box", marker_color="#7c3aed", boxmean=True), row=1, col=2)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,1)", height=350, showlegend=False, font_family="DM Sans")
            st.plotly_chart(fig, use_container_width=True)
            stats = col.describe()
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Mean", f"{stats['mean']:.4g}"); c2.metric("Std", f"{stats['std']:.4g}")
            c3.metric("Min", f"{stats['min']:.4g}"); c4.metric("Max", f"{stats['max']:.4g}")
        if cat_cols:
            cat_col = st.selectbox("Select categorical column", cat_cols, key="cat_col")
            vc = df[cat_col].value_counts().head(20)
            fig2 = px.bar(x=vc.values, y=vc.index, orientation="h", color=vc.values, color_continuous_scale="Blues", labels={"x":"Count","y":cat_col})
            fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,1)", height=300, showlegend=False, font_family="DM Sans", yaxis={"autorange":"reversed"})
            st.plotly_chart(fig2, use_container_width=True)

    with t2:
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=500, font_family="DM Sans")
            st.plotly_chart(fig, use_container_width=True)
            corr_flat = corr.where(~(corr==1.0)).unstack().dropna().reset_index()
            corr_flat.columns = ["A","B","r"]
            top = corr_flat.assign(abs_r=corr_flat["r"].abs()).sort_values("abs_r",ascending=False).head(10)
            st.dataframe(top[["A","B","r"]].reset_index(drop=True), use_container_width=True, hide_index=True)
        else:
            st.info("Need ≥2 numeric columns.")

    with t3:
        missing = df.isnull().sum()
        missing = missing[missing>0].sort_values(ascending=False)
        if len(missing)==0:
            st.success("✅ No missing values!")
        else:
            mp = (missing/len(df)*100).round(2)
            fig = px.bar(x=mp.values, y=mp.index, orientation="h", color=mp.values, color_continuous_scale="Reds", labels={"x":"Missing %","y":"Column"})
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,1)", height=max(200,len(missing)*30), font_family="DM Sans", yaxis={"autorange":"reversed"})
            st.plotly_chart(fig, use_container_width=True)

    with t4:
        if num_cols:
            st.dataframe(df[num_cols].describe().T.round(4), use_container_width=True)
        else:
            st.info("No numeric columns.")


# ── Models Tab ────────────────────────────────────────────────────────────────
def render_models_tab():
    st.markdown("<h2 style='font-size:1.5rem;'>🤖 Models & AutoML</h2>", unsafe_allow_html=True)
    if st.session_state.df is None:
        st.info("👆 Upload a dataset first."); return

    df = st.session_state.df
    all_cols = df.columns.tolist()

    c1,c2 = st.columns(2)
    with c1: target = st.selectbox("🎯 Target Column", all_cols, key="target_col")
    with c2: task_type = st.selectbox("Task Type", ["Auto-Detect","Classification","Regression","Clustering"], key="task_type")

    feature_cols = st.multiselect("Feature Columns", [c for c in all_cols if c!=target],
        default=[c for c in all_cols if c!=target][:min(15,len(all_cols)-1)], key="feature_cols")

    ca,cb,cc = st.columns(3)
    with ca:
        if st.button("🚀 Quick Train", use_container_width=True): _run_quick_train(df, target, feature_cols, task_type)
    with cb:
        if st.button("🧪 Baseline Compare", use_container_width=True): _run_baseline_models(df, target, feature_cols)
    with cc:
        if st.button("⚡ AutoGluon", use_container_width=True):
            st.warning("Install: `pip install autogluon.tabular`")

    if st.session_state.models:
        st.divider()
        st.markdown("### 🏆 Model Leaderboard")
        st.dataframe(pd.DataFrame(st.session_state.models), use_container_width=True, hide_index=True)

def _run_quick_train(df, target, feature_cols, task_type):
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

        if not feature_cols: st.error("Select features."); return
        X = df[feature_cols].copy(); y = df[target].copy()
        for col in X.select_dtypes(include="object").columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X = X.fillna(X.median(numeric_only=True))

        is_clf = (task_type=="Classification" or (task_type=="Auto-Detect" and (y.nunique()<=20 or y.dtype==object)))
        if y.dtype==object or y.dtype.name=="category": y = LabelEncoder().fit_transform(y.astype(str)); is_clf=True

        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        results = []
        models_to_run = ([("Random Forest",RandomForestClassifier(100,random_state=42)),("Logistic Reg",LogisticRegression(max_iter=1000))]
                         if is_clf else [("Random Forest",RandomForestRegressor(100,random_state=42)),("Ridge",Ridge())])
        for name, model in models_to_run:
            model.fit(X_train, y_train); preds = model.predict(X_test)
            if is_clf:
                score = round(accuracy_score(y_test,preds)*100,2)
                results.append({"Model":name,"Accuracy":f"{score}%","Task":"Classification"})
                add_agent_log("Modeler", f"{name}: Acc={score}%","success")
            else:
                r2 = round(r2_score(y_test,preds),4); mae = round(mean_absolute_error(y_test,preds),4)
                results.append({"Model":name,"R²":r2,"MAE":mae,"Task":"Regression"})
                add_agent_log("Modeler", f"{name}: R²={r2} MAE={mae}","success")
        st.session_state.models = results
        st.success(f"✅ Trained {len(results)} models!")
        st.rerun()
    except Exception as e:
        st.error(f"Training failed: {e}"); add_agent_log("Modeler",str(e),"error")

def _run_baseline_models(df, target, feature_cols):
    code = f"""# ── Baseline Model Comparison ────────────────────────────
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np, pandas as pd

X = df[{feature_cols}].copy()
y = df['{target}'].copy()
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
X = X.fillna(X.median(numeric_only=True))
if y.dtype == object: y = LabelEncoder().fit_transform(y.astype(str))

models = {{
    'RandomForest': RandomForestClassifier(100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(100, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=500),
}}
print(f"{'Model':<25} {'CV Accuracy':>12} {'Std':>8}")
print("-" * 48)
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{{name:<25}} {{scores.mean():>12.4f}} {{scores.std():>8.4f}}")
"""
    add_code_cell(code, agent="Modeler", status="pending")
    add_agent_log("Modeler","Added baseline comparison cell","info")
    st.rerun()

# ── Report Tab ────────────────────────────────────────────────────────────────
def render_report_tab():
    st.markdown("<h2 style='font-size:1.5rem;'>📄 Reports & Exports</h2>", unsafe_allow_html=True)
    if st.session_state.df is None:
        st.info("👆 Upload a dataset first."); return

    df = st.session_state.df; meta = st.session_state.metadata

    c1,c2,c3 = st.columns(3)
    with c1:
        if st.button("📝 Markdown Report", use_container_width=True):
            from utils.report import generate_markdown_report
            with st.spinner("Generating..."):
                md = generate_markdown_report(df, meta, st.session_state.models)
                st.session_state.reports.append({"type":"markdown","content":md,"ts":datetime.now().isoformat()})
                add_agent_log("Reporter","Markdown report generated","success")
            st.success("Report ready!")
    with c2:
        csv = df.to_csv(index=False).encode()
        st.download_button("⬇ Export CSV", data=csv,
            file_name=f"numdux_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", use_container_width=True)
    with c3:
        if st.session_state.code_cells and st.button("📓 Export .ipynb", use_container_width=True):
            from utils.exports import export_notebook
            nb = export_notebook(st.session_state.code_cells, meta)
            st.download_button("⬇ Download Notebook", data=nb,
                file_name=f"numdux_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb",
                mime="application/json", key="dl_nb")

    if st.session_state.reports:
        st.divider()
        for i, rep in enumerate(st.session_state.reports):
            with st.expander(f"📄 Report {i+1} — {rep['ts'][:19]}"):
                if rep["type"] == "markdown": st.markdown(rep["content"])

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()
    st.markdown("""
    <div class="numdux-header">
        <div>
            <div class="numdux-logo">🧠 Numdux</div>
            <div class="numdux-tagline">AI Data Co-Pilot · Powered by Bro AI</div>
        </div>
    </div>""", unsafe_allow_html=True)

    if st.session_state.bro_ai_running and not st.session_state.bro_ai_paused:
        col_status, col_pause = st.columns([8,1])
        with col_status:
            st.markdown("""<div style='background:linear-gradient(90deg,rgba(0,212,255,0.1),rgba(124,58,237,0.1));border:1px solid var(--accent-cyan);border-radius:8px;padding:0.6rem 1rem;margin-bottom:1rem;'>
                <span style='color:var(--accent-cyan);'>●</span>
                <span style='color:var(--text-primary);font-size:0.85rem;margin-left:8px;'><strong>Bro AI is running</strong> — Analyzing your dataset. Use sidebar to pause or send instructions.</span>
            </div>""", unsafe_allow_html=True)
        with col_pause:
            if st.button("⏸", help="Pause"): st.session_state.bro_ai_paused = True; st.rerun()

    tabs = st.tabs(["📂 Upload","🔬 Sandbox","📈 EDA","🤖 Models","📄 Report"])
    with tabs[0]: render_upload_tab()
    with tabs[1]: render_sandbox_tab()
    with tabs[2]: render_eda_tab()
    with tabs[3]: render_models_tab()
    with tabs[4]: render_report_tab()

if __name__ == "__main__":
    main()
