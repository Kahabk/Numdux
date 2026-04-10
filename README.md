# 🧠 Numdux — AI Data Co-Pilot

> **Drop a dataset. Bro AI does the rest.**

Numdux is an open-source, local-first AI co-pilot for data scientists. Upload any dataset and watch **Bro AI** automatically validate, clean, analyze, model, and report — all with full transparency, every step shown in a Jupyter-style sandbox.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **Bro AI Auto-Pilot** | Multi-agent pipeline runs automatically on upload |
| 🔬 **Jupyter-Style Sandbox** | Every AI-generated cell is visible, editable, runnable |
| 📊 **Rich EDA** | Distributions, correlations, missing value maps — interactive Plotly charts |
| 🏋️ **AutoML** | scikit-learn, GradientBoosting, cross-validation, leaderboard |
| 📄 **One-Click Reports** | Markdown + HTML reports with full analysis |
| 📓 **Export Everything** | Clean CSV, Jupyter Notebook `.ipynb`, Python script |
| 🔌 **Any LLM** | Ollama (local), OpenAI, Anthropic, Groq — via LiteLLM |
| 🔒 **Privacy-First** | All processing local by default; no data leaves your machine |
| 🐳 **Docker Ready** | One-command deployment with optional Ollama |

---

## 🚀 Quick Start

### Option 1 — Local (recommended)

```bash
# Clone
git clone https://github.com/numdux/numdux.git
cd numdux

# Install
pip install -r requirements.txt

# Run Streamlit UI
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

### Option 2 — Docker Compose (includes Ollama)

```bash
docker-compose up --build
```

Services:
- Numdux UI → http://localhost:8501  
- NumDocsX API → http://localhost:8000/docs  
- Ollama → http://localhost:11434  

Pull a model after starting:

```bash
docker exec -it numdux_ollama ollama pull llama3.2
```

---

## 🤖 LLM Configuration

Numdux supports **any LLM** via [LiteLLM](https://github.com/BerriAI/litellm):

| Provider | Config |
|----------|--------|
| **Ollama** (default, local) | Set model to e.g. `llama3.2`, ensure `ollama serve` is running |
| **OpenAI** | Select OpenAI, enter API key, choose `gpt-4o` or `gpt-4o-mini` |
| **Anthropic** | Select Anthropic, enter API key, choose `claude-sonnet-4-5` |
| **Groq** | Select Groq, enter API key, choose `llama-3.1-70b-versatile` |

**No LLM?** Numdux still works fully — it uses rule-based fallback agents that generate excellent code for all standard data science tasks.

---

## 🏗 Project Structure

```
numdux/
├── app.py                    # Main Streamlit application
│
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py       # BroAI orchestrator (LiteLLM + fallback)
│   └── graph.py              # LangGraph agent graph definition
│
├── tools/
│   ├── __init__.py
│   ├── sandbox.py            # SafeSandbox — secure code execution
│   └── data_tools.py         # File loading, profiling, type inference
│
├── utils/
│   ├── __init__.py
│   ├── report.py             # Markdown + HTML report generation
│   └── exports.py            # Notebook (.ipynb) and script export
│
├── api/
│   ├── __init__.py
│   └── main.py               # NumDocsX FastAPI backend
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🤖 Bro AI Agent Pipeline

When you upload a dataset, Bro AI orchestrates these agents:

```
Upload → Router → Validator → Cleaner → Feature Engineer
                                              ↓
                             Advisor ← Modeler ← Analyst
```

| Agent | Role |
|-------|------|
| **Router** | Decides which agents to run based on your task |
| **Validator** | Quality check: missing values, duplicates, outliers, type issues |
| **Cleaner** | Auto-cleans: dedup, impute, dtype fixes, whitespace, constant columns |
| **Feature Engineer** | Creates: polynomial features, log transforms, datetime features, z-scores |
| **Analyst** | Statistical analysis: distributions, correlations, outliers, categorical breakdown |
| **Modeler** | AutoML: trains Random Forest, Gradient Boosting, Linear models; cross-validation leaderboard |
| **Advisor** | Business insights: plain English findings, recommendations, risks |

---

## 💬 Chat with Bro AI

Use the sidebar chat to give Bro AI specific instructions:

```
"Build a churn prediction model"
"Clean this dataset for ML"
"Find the top business insights"
"Detect anomalies in the sales column"
"Create features for time-series forecasting"
"Run a full EDA and explain the results"
```

---

## 🔌 NumDocsX REST API

The FastAPI backend is available at `http://localhost:8000`:

```bash
# Upload dataset
curl -X POST http://localhost:8000/upload \
  -F "file=@your_data.csv"
# → returns session_id

# Auto-analyze
curl -X POST http://localhost:8000/auto_analyze \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess_...", "task": "full analysis"}'

# Train models
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess_...", "target_column": "churn"}'

# Generate report
curl -X POST http://localhost:8000/report \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess_...", "format": "markdown"}'
```

Interactive docs: **http://localhost:8000/docs**

---

## 📦 Optional Extras

For the full feature set:

```bash
# AutoGluon (best-in-class AutoML)
pip install autogluon.tabular

# SHAP explainability
pip install shap

# Advanced profiling
pip install ydata-profiling

# PDF report generation
pip install weasyprint

# Fast DataFrames
pip install polars
```

---

## 🛡 Sandbox Security

All AI-generated code runs in `SafeSandbox`, which:
- Blocks `open()`, `subprocess`, `os`, `socket`, `requests`
- Runs in a restricted builtins namespace
- Returns a structured result with stdout, stderr, and modified locals
- Never writes to filesystem outside temp directories

---

## 📊 Supported File Formats

| Format | Extension |
|--------|-----------|
| CSV | `.csv` |
| Tab-Separated | `.tsv` |
| Excel | `.xlsx`, `.xls` |
| Parquet | `.parquet` |
| JSON | `.json`, `.jsonl` |

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/amazing-agent`
3. Commit: `git commit -m "Add amazing new agent"`
4. Push: `git push origin feature/amazing-agent`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Built With

- [Streamlit](https://streamlit.io) — UI framework  
- [LangChain](https://langchain.com) + [LangGraph](https://langgraph.dev) — Agent orchestration  
- [LiteLLM](https://litellm.ai) — Universal LLM interface  
- [pandas](https://pandas.pydata.org) + [NumPy](https://numpy.org) — Data processing  
- [scikit-learn](https://scikit-learn.org) — Machine learning  
- [Plotly](https://plotly.com) — Visualizations  
- [FastAPI](https://fastapi.tiangolo.com) — REST API  
- [Ollama](https://ollama.ai) — Local LLM serving  

---

*Made with 🧠 by the Numdux contributors*
