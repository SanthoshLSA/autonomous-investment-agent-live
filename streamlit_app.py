"""
Streamlit Web Dashboard for the Autonomous Investment Research Agent.
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Investment Agent Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Imports from src package
from src.agents.graph import (
    build_investment_graph,
    resume_with_approval,
    run_investment_analysis,
)
from src.config import get_api_keys, get_config
from src.portfolio.backtester import Backtester
from src.portfolio.simulator_db import (
    create_user,
    execute_trade,
    get_daily_snapshots,
    get_user_cash,
    get_user_holdings,
    get_user_transactions,
    init_db,
    reset_user_portfolio,
    verify_user,
)
from src.reports.charts import (
    build_allocation_pie,
    build_price_candlestick,
    build_risk_heatmap,
)
from src.reports.generator import generate_daily_report

# Initialize core variables
config = get_config()
api_keys = get_api_keys()

# Initialize DB tables
try:
    init_db()
except Exception as e:
    st.error(f"Database Connection Error: {str(e)}")

# Ensure keys exist in session state to handle page refreshes
if "graph" not in st.session_state:
    st.session_state.graph = build_investment_graph(config, api_keys)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"dashboard-{datetime.now().strftime('%Y%m%d%H%M%S')}"
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "approval_pending" not in st.session_state:
    st.session_state.approval_pending = False
if "report_paths" not in st.session_state:
    st.session_state.report_paths = None
if "authenticated_user" not in st.session_state:
    if "user" in st.query_params:
        st.session_state.authenticated_user = st.query_params["user"]
    else:
        st.session_state.authenticated_user = None
if "selected_tickers" not in st.session_state:
    st.session_state.selected_tickers = []


# Custom CSS to match santhoshlsa.vercel.app visual layout
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Outfit:wght@300;400;600;700&display=swap');

/* Hide default Streamlit elements and sidebar */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stHeader"] {background: transparent !important;}
[data-testid="stSidebar"] {display: none !important;}
[data-testid="stSidebarCollapsedControl"] {display: none !important;}

/* Main layout background & fonts */
.stApp {
    background: radial-gradient(circle at 10% 20%, rgba(168, 85, 247, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(56, 189, 248, 0.08) 0%, transparent 40%),
                #07040d !important;
    font-family: 'Outfit', sans-serif !important;
    color: #f8fafc !important;
}

/* Headings styling */
h1, h2, h3, h4, h5, h6, [data-testid="stHeader"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    text-transform: uppercase !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0c0816 !important;
    border-right: 1px solid rgba(168, 85, 247, 0.15) !important;
}

/* Buttons styling */
.stButton>button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100% !important;
}

/* Primary Button Styling */
.stButton>button[kind="primary"] {
    background: linear-gradient(135deg, #a855f7 0%, #6366f1 100%) !important;
    color: white !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    box-shadow: 0 0 15px rgba(168, 85, 247, 0.25) !important;
}
.stButton>button[kind="primary"]:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 0 25px rgba(168, 85, 247, 0.45) !important;
    border-color: rgba(168, 85, 247, 0.5) !important;
}

/* Secondary Button Styling */
.stButton>button[kind="secondary"] {
    background: rgba(12, 8, 22, 0.6) !important;
    color: #cbd5e1 !important;
    border: 1px solid rgba(168, 85, 247, 0.2) !important;
}
.stButton>button[kind="secondary"]:hover {
    background: rgba(168, 85, 247, 0.15) !important;
    color: #a855f7 !important;
    border-color: #a855f7 !important;
}

/* Glassmorphic Cards */
div[data-testid="stMetric"], .stAlert {
    background: rgba(12, 8, 22, 0.6) !important;
    border: 1px solid rgba(168, 85, 247, 0.15) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4) !important;
}

/* Inputs styling */
div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input, div[data-baseweb="select"] {
    background-color: rgba(12, 8, 22, 0.85) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(168, 85, 247, 0.25) !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
}
div[data-testid="stTextInput"] input:focus, div[data-testid="stNumberInput"] input:focus {
    border-color: #a855f7 !important;
    box-shadow: 0 0 10px rgba(168, 85, 247, 0.3) !important;
}

/* Selectbox / Multiselect styling */
div[data-baseweb="select"] {
    background-color: rgba(12, 8, 22, 0.8) !important;
    border-radius: 12px !important;
}

/* Expanders styling */
div[data-testid="stExpander"] {
    background: rgba(12, 8, 22, 0.5) !important;
    border: 1px solid rgba(168, 85, 247, 0.15) !important;
    border-radius: 12px !important;
}

/* Table / Dataframe styling */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(168, 85, 247, 0.15) !important;
    border-radius: 12px !important;
    background: rgba(12, 8, 22, 0.6) !important;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    background-color: transparent !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    background-color: rgba(255, 255, 255, 0.01) !important;
    border: 1px solid rgba(255, 255, 255, 0.03) !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
}

.stTabs [aria-selected="true"] {
    color: #a855f7 !important;
    border-bottom: 2px solid #a855f7 !important;
    background-color: rgba(168, 85, 247, 0.1) !important;
}

/* Text elements custom highlights */
.stMarkdown p {
    color: #cbd5e1 !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
}

/* Gradient heading highlight */
.gradient-header {
    background: linear-gradient(to right, #a855f7, #ffffff, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 800;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ── Title & Sidebar ──────────────────────────────────────────────────────────
st.markdown(
    '<h1 class="gradient-header">AUTONOMOUS INVESTMENT ADVISER DESK</h1>', unsafe_allow_html=True
)
st.markdown("---")


# ── Global Authentication Gate ───────────────────────────────────────────────
if st.session_state.authenticated_user is None:
    st.info(
        "Welcome! Please sign in or register a new account to unlock the Investment Adviser Desk."
    )

    col_auth1, col_auth2 = st.columns(2)
    with col_auth1:
        st.markdown("#### Sign In")
        gate_username = st.text_input("Username", key="gate_user_field").strip()
        gate_password = st.text_input("Password", type="password", key="gate_pass_field")
        if st.button("Sign In", key="gate_signin_btn", type="primary", use_container_width=True):
            if verify_user(gate_username, gate_password):
                st.session_state.authenticated_user = gate_username
                st.query_params["user"] = gate_username
                st.rerun()
            else:
                st.error("Invalid username or password.")
    with col_auth2:
        st.markdown("#### Create Account")
        reg_username = st.text_input("New Username", key="gate_reg_user_field").strip()
        reg_password = st.text_input("New Password", type="password", key="gate_reg_pass_field")
        if st.button("Register Account", key="gate_register_btn", use_container_width=True):
            if not reg_username or not reg_password:
                st.error("Username and Password cannot be empty.")
            elif create_user(reg_username, reg_password):
                st.success("Account created successfully! Please Sign In.")
            else:
                st.error("Username taken or database error.")

    st.stop()


# ── Unified Control Board Header ──────────────────────────────────────────────
col_prof1, col_prof2 = st.columns([3, 1])
with col_prof1:
    st.write(
        f"Stance: **{config.portfolio.risk_tolerance.upper()}** | Optimization: **{config.portfolio.optimization_method}**"
    )
with col_prof2:
    st.markdown(f"Logged in: **{st.session_state.authenticated_user}**")
    if st.button("Sign Out", key="top_signout_btn", use_container_width=True):
        st.session_state.authenticated_user = None
        st.query_params.clear()
        st.rerun()

st.markdown("---")

# ── Main Workspace Navigation ────────────────────────────────────────────────
st.markdown("#### Navigation")
workspace_page = st.radio(
    "Choose Workspace",
    ["AI Adviser Desk", "Paper Trading Workspace"],
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("---")

# ── Paper Trading Workspace Rendering ─────────────────────────────────────────
if workspace_page == "Paper Trading Workspace":
    col_title, col_refresh = st.columns([3, 1])
    with col_title:
        st.markdown("### Paper Trading Simulator")
    with col_refresh:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button(
            "Refresh Data", use_container_width=True, type="secondary", key="trading_refresh_btn"
        ):
            st.rerun()
    username = st.session_state.authenticated_user

    # 1. Fetch current portfolio parameters
    cash = get_user_cash(username)
    holdings = get_user_holdings(username)

    # Calculate valuations
    from src.portfolio.valuation_engine import get_latest_price

    holdings_value = 0.0
    holdings_cost = 0.0

    holdings_data = []
    for pos in holdings:
        ticker = pos["ticker"]
        shares = pos["shares"]
        cost = pos["avg_cost"]

        # Get live price
        price = get_latest_price(ticker)
        if price <= 0:
            price = cost

        mkt_val = shares * price
        pos_cost = shares * cost
        pnl = mkt_val - pos_cost
        pnl_pct = (pnl / pos_cost * 100) if pos_cost > 0 else 0.0

        holdings_value += mkt_val
        holdings_cost += pos_cost

        holdings_data.append(
            {
                "Asset": ticker,
                "Shares Owned": f"{shares:.4f}",
                "Avg Purchase Price (INR)": f"{cost:,.2f} INR",
                "Current Live Price (INR)": f"{price:,.2f} INR",
                "Total Value (INR)": f"{mkt_val:,.2f} INR",
                "Unrealized P&L (INR)": f"{pnl:+,.2f} INR ({pnl_pct:+.2f}%)",
            }
        )

    total_value = cash + holdings_value
    net_pnl = total_value - 1000000.0  # Initial principal is 10 Lakh
    pnl_pct = (net_pnl / 1000000.0) * 100

    # 2. Render Metric cards
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Cash Balance", f"{cash:,.2f} INR")
    col_m2.metric("Portfolio Value", f"{total_value:,.2f} INR")
    col_m3.metric("Total Invested", f"{holdings_cost:,.2f} INR")
    col_m4.metric("Unrealized P&L", f"{net_pnl:+,.2f} INR ({pnl_pct:+.2f}%)")
    st.markdown("---")

    # Rebalance to AI Recommendations Button
    if st.session_state.analysis_results:
        state_res = st.session_state.analysis_results
        recomm = state_res.get("portfolio_recommendation", {})
        allocations = recomm.get("allocations", {})

        if allocations and st.button(
            "Apply AI Recommended Allocations Directly",
            type="primary",
            use_container_width=True,
        ):
            # Calculate required trades to match recommended allocations
            trades_executed = []
            errors_encountered = []

            # We need to map current holdings: {ticker: shares}
            current_holdings_dict = {pos["ticker"]: pos["shares"] for pos in holdings}

            # Fetch live prices for all recommended assets
            from src.portfolio.valuation_engine import get_latest_price

            sell_trades = []
            buy_trades = []

            for ticker, weight in allocations.items():
                if ticker.startswith("^") or ticker.upper() == "CASH":
                    continue

                price = get_latest_price(ticker)
                if price <= 0:
                    continue

                target_val = total_value * weight
                target_shares = target_val / price
                current_shares = current_holdings_dict.get(ticker, 0.0)
                diff_shares = target_shares - current_shares

                if diff_shares < -0.001:
                    sell_trades.append((ticker, abs(diff_shares), price))
                elif diff_shares > 0.001:
                    buy_trades.append((ticker, diff_shares, price))

            # Execute sells first
            for ticker, qty, price in sell_trades:
                res = execute_trade(username, ticker, "SELL", qty, price)
                if res == "SUCCESS":
                    trades_executed.append(f"Sold {qty:.4f} shares of {ticker} at {price:,.2f} INR")
                else:
                    errors_encountered.append(f"Failed to sell {ticker}: {res}")

            # Re-fetch cash after sells to verify buying power
            cash_after_sells = get_user_cash(username)

            # Execute buys
            for ticker, qty, price in buy_trades:
                cost = qty * price
                if cash_after_sells >= cost:
                    res = execute_trade(username, ticker, "BUY", qty, price)
                    if res == "SUCCESS":
                        trades_executed.append(
                            f"Bought {qty:.4f} shares of {ticker} at {price:,.2f} INR"
                        )
                        cash_after_sells -= cost
                    else:
                        errors_encountered.append(f"Failed to buy {ticker}: {res}")
                else:
                    max_qty = cash_after_sells / price
                    if max_qty > 0.001:
                        res = execute_trade(username, ticker, "BUY", max_qty, price)
                        if res == "SUCCESS":
                            trades_executed.append(
                                f"Bought {max_qty:.4f} shares of {ticker} (partial order) at {price:,.2f} INR"
                            )
                            cash_after_sells -= max_qty * price
                        else:
                            errors_encountered.append(f"Failed to buy {ticker}: {res}")

            if trades_executed:
                st.success(
                    "Portfolio rebalanced to align with AI Recommended Allocations! Executed actions:\n"
                    + "\n".join([f"- {t}" for t in trades_executed])
                )
            if errors_encountered:
                st.error(
                    "Some orders encountered issues during execution:\n"
                    + "\n".join([f"- {e}" for e in errors_encountered])
                )

            st.rerun()

    # 3. Main Dashboard grid
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("#### Portfolio Holdings")
        if holdings_data:
            st.dataframe(pd.DataFrame(holdings_data), use_container_width=True, hide_index=True)
        else:
            st.info(
                "No active holdings inside your portfolio. Execute orders on the right to populate your holdings."
            )

        # Daily Snapshot Progression Line Chart
        st.markdown("#### Portfolio Progression")
        snapshots = get_daily_snapshots(username)
        if snapshots:
            snap_df = pd.DataFrame(snapshots)
            snap_df["date"] = pd.to_datetime(snap_df["date"])

            import plotly.express as px

            fig = px.area(
                snap_df,
                x="date",
                y="total_value",
                title="Historical Portfolio Value (INR)",
                labels={"date": "Date", "total_value": "Portfolio Value (INR)"},
                color_discrete_sequence=["#a855f7"],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show daily summary change
            last_snap = snapshots[-1]
            st.success(
                f"**Market Daily Snapshot Update**: Your daily PnL changed by `{last_snap['daily_pnl']:+,.2f} INR` on {last_snap['date']}."
            )
        else:
            st.info(
                "Portfolio progression chart will populate automatically after your first daily closing run."
            )

    with col_right:
        # Order Execution Form
        st.markdown("#### Execute Order")
        selected_ticker = st.selectbox("Asset Ticker", config.watchlist.all_tickers)
        action = st.selectbox("Order Type", ["BUY", "SELL"])
        shares = st.number_input(
            "Share Quantity", min_value=0.0001, step=1.0, format="%.4f", value=1.0
        )

        # Fetch live price to display estimated value
        live_price = get_latest_price(selected_ticker)
        est_value = shares * live_price

        st.write(f"Estimated Price: **{live_price:,.2f} INR**")
        st.write(f"Estimated Order Value: **{est_value:,.2f} INR**")

        if st.button("Submit Order", type="primary", use_container_width=True):
            trade_result = execute_trade(username, selected_ticker, action, shares, live_price)
            if trade_result == "SUCCESS":
                st.success(f"Order executed successfully: {action} {shares} {selected_ticker}")
                st.rerun()
            else:
                st.error(trade_result)

        # Transaction History Expandable Log
        with st.expander("Transaction Log"):
            txs = get_user_transactions(username)
            if txs:
                st.dataframe(pd.DataFrame(txs), use_container_width=True, hide_index=True)
            else:
                st.write("No transactions recorded yet.")

        # Danger Zone Reset Button
        st.markdown("---")
        st.markdown("#### Danger Zone")
        confirm_reset = st.checkbox("Confirm reset portfolio to default 10 Lakh")
        if st.button(
            "Reset Portfolio",
            type="secondary",
            disabled=not confirm_reset,
            use_container_width=True,
        ):
            if reset_user_portfolio(username):
                st.success("Portfolio successfully reset back to default values!")
                st.rerun()
            else:
                st.error("Failed to reset portfolio.")

    st.stop()


# ── AI Adviser Workspace Rendering ────────────────────────────────────────────
st.markdown("### AI Adviser Desk")

# Stance & Optimization explanation expander
with st.expander("Understanding Configuration Parameters"):
    st.markdown("""
    - **Stance (Risk Tolerance)**: **`CONSERVATIVE`** instructs the adviser agent to prioritize capital preservation over high returns. It focuses on stable assets with historically low variance.
    - **Optimization Method**: **`MIN VOLATILITY`** calculates asset weight allocations mathematically designed to minimize the expected variance of the overall portfolio. It relies on historical price covariance to minimize portfolio fluctuations and drawdowns.
    """)

st.markdown("#### Watchlist Assets Selection")

# Interactive button grid for asset selection
asset_categories = {
    "🇺🇸 US Stocks": config.watchlist.us_stocks,
    "🇮🇳 Indian Stocks": config.watchlist.indian_stocks,
    "🪙 Crypto": config.watchlist.crypto,
    "📊 Indices": config.watchlist.indices
}

for cat_name, tickers in asset_categories.items():
    st.markdown(f"**{cat_name}**")
    cols = st.columns(5)
    for idx, ticker in enumerate(tickers):
        col = cols[idx % 5]
        is_selected = ticker in st.session_state.selected_tickers
        btn_label = f"✓ {ticker}" if is_selected else ticker
        if col.button(btn_label, key=f"btn_select_{ticker}", use_container_width=True, type="primary" if is_selected else "secondary"):
            if ticker in st.session_state.selected_tickers:
                st.session_state.selected_tickers.remove(ticker)
            else:
                st.session_state.selected_tickers.append(ticker)
            st.rerun()

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
col_sub1, col_sub2, col_sub3 = st.columns([1.5, 1.5, 5])
with col_sub1:
    if st.button("Select Invested", use_container_width=True, key="btn_select_invested_adviser"):
        holdings = get_user_holdings(st.session_state.authenticated_user)
        st.session_state.selected_tickers = [pos["ticker"] for pos in holdings]
        st.rerun()
with col_sub2:
    if st.button("Clear All", use_container_width=True, key="btn_clear_all_adviser"):
        st.session_state.selected_tickers = []
        st.rerun()
with col_sub3:
    st.markdown(f"Selected: **{', '.join(st.session_state.selected_tickers) if st.session_state.selected_tickers else 'None'}**")

selected_tickers = st.session_state.selected_tickers
st.markdown("---")

col_act1, col_act2 = st.columns(2)
with col_act1:
    if st.button(
        "Run Full Market Scan", type="primary", use_container_width=True, key="main_run_scan"
    ):
        if not selected_tickers:
            st.error("Please select at least one asset to scan.")
        else:
            with st.spinner("Executing sequential multi-agent LangGraph scan..."):
                try:
                    st.session_state.graph = build_investment_graph(config, get_api_keys())
                    results = run_investment_analysis(
                        st.session_state.graph, selected_tickers, st.session_state.thread_id
                    )
                    st.session_state.analysis_results = results
                    st.session_state.approval_pending = True
                    st.success("Scan completed. Awaiting portfolio allocation approval.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Execution Error: {str(e)}")

with col_act2:
    if st.button("Reset Session ID", use_container_width=True, key="main_reset_session"):
        st.session_state.thread_id = f"dashboard-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        st.session_state.analysis_results = None
        st.session_state.approval_pending = False
        st.success("Session ID re-generated.")
        st.rerun()

st.markdown("---")

if not st.session_state.analysis_results:
    st.info(
        "No active scan results. Configure watchlist assets above and click 'Run Full Market Scan' to begin analysis."
    )


# ── Human Approval Gate UI ──────────────────────────────────────────────────
if st.session_state.approval_pending and st.session_state.analysis_results:
    st.info("**LangGraph approval interrupt reached**: Review allocations below before proceeding.")
    col_app1, col_app2 = st.columns(2)

    with col_app1:
        if st.button("Approve & Generate Reports", type="primary", use_container_width=True):
            with st.spinner("Resuming workflow pipeline..."):
                try:
                    final_state = resume_with_approval(
                        st.session_state.graph, st.session_state.thread_id, "approve"
                    )
                    # Generate daily report HTML
                    rep_paths = generate_daily_report(final_state)
                    st.session_state.report_paths = rep_paths

                    st.session_state.analysis_results = final_state
                    st.session_state.approval_pending = False
                    st.success("Approved! Daily HTML and PDF reports generated successfully.")
                except Exception as e:
                    st.error(f"Error resuming graph: {str(e)}")

    with col_app2:
        if st.button("Reject & Stop Pipeline", type="secondary", use_container_width=True):
            try:
                resume_with_approval(st.session_state.graph, st.session_state.thread_id, "reject")
                st.session_state.approval_pending = False
                st.session_state.analysis_results = None
                st.warning("Recommendation rejected. Execution halted.")
            except Exception as e:
                st.error(f"Error: {str(e)}")


# ── Render Analytics Panels ──────────────────────────────────────────────────
if st.session_state.analysis_results:
    state = st.session_state.analysis_results

    # Show error logs for debugging
    errors = state.get("error_log", [])
    if errors:
        with st.expander("System Debug & Connection Logs"):
            for err in errors:
                st.error(err)
        st.markdown("---")

    # Downloadable PDF Report Button
    if st.session_state.report_paths and "pdf" in st.session_state.report_paths:
        pdf_path = st.session_state.report_paths["pdf"]
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="Download PDF Report Document",
                data=pdf_bytes,
                file_name=f"Daily_Investment_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Target Allocations", "Asset Analysis", "Backtesting Validation"])

    # 1. Allocation Page
    with tab1:
        recomm = state.get("portfolio_recommendation", {})
        if recomm:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Asset Breakdown")
                fig_pie = build_allocation_pie(recomm.get("allocations", {}))
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                st.subheader("Advisor Stance Summary")
                st.info(recomm.get("portfolio_summary", "No summary text."))

                st.subheader("Key Caveats / Warnings")
                for w in recomm.get("warnings", []):
                    st.warning(w)

            st.subheader("Execution Rationale per Asset")
            for ticker, text in recomm.get("rationale", {}).items():
                st.markdown(f"**{ticker}**: {text}")
        else:
            st.warning("No portfolio recommendations calculated.")

    # 2. Deep Dive Page
    with tab2:
        composite = state.get("composite_scores", {})
        if composite:
            fig_heat = build_risk_heatmap(composite)
            st.plotly_chart(fig_heat, use_container_width=True)

            selected_ticker = st.selectbox("Deep Dive Stock Ticker", list(composite.keys()))

            # Candlestick
            market = state.get("market_data", {}).get(selected_ticker, {})
            if market and market.get("prices"):
                fig_candle = build_price_candlestick(selected_ticker, market["prices"])
                st.plotly_chart(fig_candle, use_container_width=True)

                # News
                st.subheader("Sentiment Source News Headlines")
                for article in market.get("news", []):
                    st.markdown(
                        f"- **[{article.get('source')}]** [{article.get('title')}]({article.get('url')})"
                    )
        else:
            st.warning("Run a scan to check asset metrics.")

    # 3. Backtesting Page
    with tab3:
        st.subheader("Historical Backtest (5 Years)")
        st.write("Simulating calculated allocation against S&P 500 benchmark...")

        # Load price df from state to run backtest
        market = state.get("market_data", {})
        if market:
            price_series_dict = {}
            for ticker, bundle in market.items():
                prices = bundle.get("prices", [])
                if prices:
                    pdf = pd.DataFrame(prices)
                    pdf["date"] = pd.to_datetime(pdf["date"], utc=True)
                    pdf.set_index("date", inplace=True)
                    pdf.sort_index(inplace=True)
                    price_series_dict[ticker] = pdf["close"]

            prices_df = pd.DataFrame(price_series_dict).ffill().bfill()

            if not prices_df.empty:
                optimizer_weights = state.get("portfolio_recommendation", {}).get("allocations", {})

                bt = Backtester(config.backtest)
                metrics = bt.run_backtest(prices_df, optimizer_weights)

                if "error" not in metrics:
                    col_bt1, col_bt2, col_bt3 = st.columns(3)
                    col_bt1.metric("CAGR Return", f"{metrics['annual_return']:.2%}")
                    col_bt2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                    col_bt3.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")

                    st.write(f"Benchmark Cumulative Return: **{metrics['benchmark_return']:.2%}**")
                    st.write(f"Strategy Cumulative Return: **{metrics['total_return']:.2%}**")
                else:
                    st.error(f"Backtesting error: {metrics['error']}")
            else:
                st.warning("Insufficient prices dataframe generated.")
        else:
            st.warning("No market data to backtest.")
