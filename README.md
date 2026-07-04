# Autonomous Investment Research Agent

An enterprise-grade, production-ready Multi-Agent Investment Adviser Desk built using Python 3.11+, LangGraph, and PyPortfolioOpt. Deploys containerized and operates fully automated.

## 🚀 Key Architectural Features
- **Data Layer**: Aggregate ticker histories and business news from multiple data providers (yfinance + Finnhub + NewsAPI). Caches locally using a thread-safe SQLite backend.
- **Analysis Engine**: Calculates technical indicators (RSI, MACD, Moving Averages), risk variables (historical/parametric VaR, CVaR Expected Shortfall, drawdown metrics, Sharpe ratios), and sentiment indexes (VADER + exponential decay time weighting).
- **AI Agent Core**: LangGraph sequential multi-agent workflow (Researcher Agent -> Analyst Agent -> Recommender Agent) that halts execution at a **Human-in-the-Loop approval gate** for allocation inspection.
- **Portfolio Optimization**: Optimizes weights using `PyPortfolioOpt` with support for minimum volatility or maximum Sharpe targets. Backtests portfolio strategies over 5+ years of historical data with drift-rebalancing order calculations.
- **Visual Dashboards**: Streamlit dashboard visualization tracking allocations, risk heatmap matrixes, equity curves, and generated HTML daily reports.

---

## 🛠️ Local Development Setup

### 1. Requirements
Ensure Python 3.11 or higher and the `uv` package manager are installed.

### 2. Configure Credentials
Duplicate `.env.example` to `.env` and fill in your API tokens:
```bash
cp .env.example .env
```
Register free accounts to fetch keys:
- **NewsAPI Key**: [newsapi.org](https://newsapi.org/)
- **Finnhub Key**: [finnhub.io](https://finnhub.io/)

### 3. Install Dependencies
Synchronize project packages using the `uv` CLI tool:
```bash
uv sync
```

### 4. Run the Streamlit Web App
Launch the interactive web portal locally:
```bash
uv run streamlit run streamlit_app.py
```

### 5. Run Scheduled CLI Scans
Trigger an immediate scan from command-line mode:
```bash
uv run Python src/main.py run
```
Or start the daily APScheduler daemon:
```bash
uv run Python src/main.py daemon
```
*Calculated HTML reports will be outputted under `reports/output/<YYYY-MM-DD>/report.html`.*

---

## 🐋 Docker Containerized Execution
Build and launch the web server using Docker:
```bash
docker build -t investment-agent .
docker run -p 8501:8501 --env-file .env investment-agent
```
Access the dashboard at `http://localhost:8501`.
