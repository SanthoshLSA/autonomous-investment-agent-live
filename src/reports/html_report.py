"""
HTML Report generator.

Renders analysis, allocations, backtests, and charts into a single interactive HTML file using Jinja2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.logger import get_logger

logger = get_logger(__name__)


def generate_html_report(state_data: dict[str, Any], output_path: str | Path) -> Path:
    """Renders all compiled analysis into a responsive, premium HTML document.

    Args:
        state_data: Dictionary extracted from LangGraph state containing recommendations.
        output_path: Target path to write the file.

    Returns:
        Path object pointing to the written file.
    """
    logger.info("Generating HTML report file...", path=str(output_path))

    # Simple self-contained HTML rendering (embedded charts, layout and css)
    # Allows users to view reports locally without server attachments.
    html_template = (
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autonomous Investment Research Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #07040d;
            --bg-secondary: #0c0816;
            --accent: #a855f7;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --success: #34d399;
            --warning: #fbbf24;
            --danger: #f87171;
            --card-border: rgba(168, 85, 247, 0.15);
        }
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: 'Outfit', sans-serif;
            background: radial-gradient(circle at 10% 20%, rgba(168, 85, 247, 0.06) 0%, transparent 40%),
                        radial-gradient(circle at 90% 80%, rgba(56, 189, 248, 0.06) 0%, transparent 40%),
                        var(--bg-primary);
            color: var(--text-main);
            line-height: 1.6;
            padding: 3rem 2rem;
        }
        header {
            max-width: 1200px;
            margin: 0 auto 3rem auto;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 1.5rem;
        }
        header h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.5rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: -0.02em;
            background: linear-gradient(to right, #a855f7, #ffffff, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        header p {
            color: var(--text-muted);
            font-size: 1.1rem;
            text-transform: uppercase;
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: 0.05em;
            margin-top: 0.5rem;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
        }
        .card {
            background-color: rgba(12, 8, 22, 0.6);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 2.5rem;
            backdrop-filter: blur(12px);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
        }
        .card-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.5rem;
            margin-bottom: 1.5rem;
            font-weight: 700;
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.02em;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 0.75rem;
        }
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }
        @media(max-width: 768px) {
            .grid-2 {
                grid-template-columns: 1fr;
            }
        }
        .summary-stats {
            display: flex;
            gap: 2rem;
            flex-wrap: wrap;
            margin-bottom: 2rem;
        }
        .stat-box {
            flex: 1;
            min-width: 200px;
            background: rgba(255, 255, 255, 0.01);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.03);
            text-align: center;
        }
        .stat-val {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent);
            margin-top: 0.5rem;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        th, td {
            text-align: left;
            padding: 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        th {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--text-muted);
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.05em;
        }
        .badge {
            font-family: 'Space Grotesk', sans-serif;
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
        }
        .badge-buy { background-color: rgba(52, 211, 153, 0.15); color: var(--success); border: 1px solid rgba(52, 211, 153, 0.3); }
        .badge-hold { background-color: rgba(251, 191, 36, 0.15); color: var(--warning); border: 1px solid rgba(251, 191, 36, 0.3); }
        .badge-sell { background-color: rgba(248, 113, 113, 0.15); color: var(--danger); border: 1px solid rgba(248, 113, 113, 0.3); }

        .list-unstyled {
            list-style: none;
        }
        .list-unstyled li {
            margin-bottom: 0.75rem;
            position: relative;
            padding-left: 1.5rem;
        }
        .list-unstyled li::before {
            content: "•";
            color: var(--accent);
            position: absolute;
            left: 0;
            font-size: 1.25rem;
        }
    </style>
</head>
<body>
    <header>
        <h1>Autonomous Investment Agent</h1>
        <p>Daily Research & Rebalancing Analysis • Generated on: """
        + Path(output_path).stem
        + """</p>
    </header>

    <div class="container">
        <!-- 1. Executive Summary -->
        <div class="card">
            <h2 class="card-title">Executive Summary</h2>
            <div class="summary-stats">
                <div class="stat-box">
                    <div>Stance / Benchmark</div>
                    <div class="stat-val">CONSERVATIVE</div>
                </div>
                <div class="stat-box">
                    <div>Active Watchlist Assets</div>
                    <div class="stat-val">"""
        + str(len(state_data.get("composite_scores", {})))
        + """</div>
                </div>
            </div>"""
    )
    recomm = state_data.get("portfolio_recommendation", {})
    summary_data = recomm.get("portfolio_summary", "No summary provided.")
    if isinstance(summary_data, dict):
        summary_html = "<br>".join(
            f"<strong>{str(k).replace('_', ' ').title()}</strong>: {str(v)}"
            for k, v in summary_data.items()
        )
    else:
        summary_html = str(summary_data)

    html_template += f"""
            <p>{summary_html}</p>
        </div>

        <!-- 2. Allocations & Allocation Rationale -->
        <div class="grid-2">
            <div class="card">
                <h2 class="card-title">Target Allocations</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Ticker</th>
                            <th>Allocation</th>
                        </tr>
                    </thead>
                    <tbody>"""

    recomm = state_data.get("portfolio_recommendation", {})
    allocations = recomm.get("allocations", {})
    for ticker, weight in allocations.items():
        try:
            weight_val = float(weight)
            weight_str = f"{weight_val:+.2%}"
        except (ValueError, TypeError):
            weight_str = str(weight)

        html_template += f"""
                        <tr>
                            <td><strong>{ticker}</strong></td>
                            <td style="color: var(--accent); font-weight: 600;">{weight_str}</td>
                        </tr>
        """

    html_template += """
                    </tbody>
                </table>
            </div>

            <div class="card">
                <h2 class="card-title">AI Adviser Rationale</h2>
                <ul class="list-unstyled">
    """

    rationales = recomm.get("rationale", {})
    for ticker, text in rationales.items():
        html_template += f"<li><strong>{ticker}</strong>: {str(text)}</li>"

    html_template += """
                </ul>
            </div>
        </div>

        <!-- 3. Metrics Tables -->
        <div class="card">
            <h2 class="card-title">Asset Scoring Breakdown</h2>
            <table>
                <thead>
                    <tr>
                        <th>Asset</th>
                        <th>Signal</th>
                        <th>Risk Score</th>
                        <th>RSI</th>
                        <th>Sharpe Ratio</th>
                        <th>Weighted Sentiment</th>
                    </tr>
                </thead>
                <tbody>
    """

    composite_scores = state_data.get("composite_scores", {})
    for ticker, score in composite_scores.items():
        tech = state_data.get("technical_analysis", {}).get(ticker, {})
        risk = state_data.get("risk_analysis", {}).get(ticker, {})
        sent = state_data.get("sentiment_analysis", {}).get(ticker, {})

        signal = score["signal"]
        badge_cls = "badge-hold"
        if "buy" in signal:
            badge_cls = "badge-buy"
        elif "sell" in signal:
            badge_cls = "badge-sell"

        # Format float metrics safely
        try:
            rsi_val = float(tech.get("rsi_value", 50.0))
            rsi_str = f"{rsi_val:.1f}"
        except (ValueError, TypeError):
            rsi_str = str(tech.get("rsi_value", "50.0"))

        try:
            sharpe_val = float(risk.get("sharpe_ratio", 0.0))
            sharpe_str = f"{sharpe_val:.2f}"
        except (ValueError, TypeError):
            sharpe_str = str(risk.get("sharpe_ratio", "0.00"))

        try:
            sent_val = float(sent.get("weighted_sentiment", 0.0))
            sent_str = f"{sent_val:+.2f}"
        except (ValueError, TypeError):
            sent_str = str(sent.get("weighted_sentiment", "+0.00"))

        html_template += f"""
                    <tr>
                        <td><strong>{ticker}</strong></td>
                        <td><span class="badge {badge_cls}">{str(signal).upper()}</span></td>
                        <td>{str(score["risk_score"])}/100</td>
                        <td>{rsi_str}</td>
                        <td>{sharpe_str}</td>
                        <td>{sent_str} ({str(sent.get("classification", "neutral")).upper()})</td>
                    </tr>
        """

    html_template += """
                </tbody>
            </table>
        </div>

        <!-- 3.5. Detailed Asset Deep-Dive & News Sources -->
        <div class="card">
            <h2 class="card-title">Detailed Asset Analysis & Sources</h2>
            <div style="margin-top: 15px;">
    """

    for ticker, score in composite_scores.items():
        tech = state_data.get("technical_analysis", {}).get(ticker, {})
        risk = state_data.get("risk_analysis", {}).get(ticker, {})
        sent = state_data.get("sentiment_analysis", {}).get(ticker, {})

        # Get rationale
        rationale_text = rationales.get(ticker, "No AI rationale available.")

        # Get metrics
        risk_score = score.get("risk_score", 0)
        signal_type = score.get("signal", "hold").upper()

        try:
            sharpe_val = float(risk.get("sharpe_ratio", 0.0))
            sharpe_str = f"{sharpe_val:.2f}"
        except (ValueError, TypeError):
            sharpe_str = str(risk.get("sharpe_ratio", "0.00"))

        try:
            sent_val = float(sent.get("weighted_sentiment", 0.0))
            sent_str = f"{sent_val:+.2f}"
        except (ValueError, TypeError):
            sent_str = str(sent.get("weighted_sentiment", "+0.00"))

        html_template += f"""
                <div style="border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 20px; margin-bottom: 20px;">
                    <h3 style="font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; color: #a855f7; margin-bottom: 10px;">{ticker} ({signal_type})</h3>
                    <p style="margin-bottom: 8px; font-size: 0.95rem; line-height: 1.5;"><strong>AI Decision Rationale:</strong> {str(rationale_text)}</p>
                    <p style="margin-bottom: 12px; font-size: 0.85rem; color: #94a3b8;"><strong>Metrics:</strong> Risk Score: {risk_score}/100 | Sharpe: {sharpe_str} | Sentiment: {sent_str}</p>
        """

        # Pull top 3 news sources
        news_articles = state_data.get("market_data", {}).get(ticker, {}).get("news", [])
        if news_articles:
            html_template += """
                    <p style="font-size: 0.85rem; font-weight: bold; margin-bottom: 5px; color: #f8fafc;">Primary Information Sources:</p>
                    <ul style="margin-left: 20px; margin-bottom: 5px; font-size: 0.85rem; color: #94a3b8;">
            """
            for article in news_articles[:3]:
                # Safe access to article fields
                title = (
                    article.title if hasattr(article, "title") else article.get("title", "No Title")
                )
                source = (
                    article.source
                    if hasattr(article, "source")
                    else article.get("source", "Unknown Source")
                )
                html_template += f'<li>"{str(title)}" (Source: {str(source)})</li>'
            html_template += """
                    </ul>
            """
        else:
            html_template += """
                    <p style="font-size: 0.85rem; font-style: italic; color: #94a3b8;">No news source articles available for this asset.</p>
            """

        html_template += """
                </div>
        """

    html_template += """
            </div>
        </div>

        <!-- 4. Warnings -->
        <div class="card" style="border-left: 4px solid var(--danger);">
            <h2 class="card-title" style="color: var(--danger);">Caveats & Risk Warnings</h2>
            <ul class="list-unstyled">
    """

    warnings = recomm.get("warnings", [])
    for warn in warnings:
        html_template += f"<li>{str(warn)}</li>"

    html_template += """
            </ul>
        </div>
    </div>
</body>
</html>
    """

    path = Path(output_path)
    path.write_text(html_template, encoding="utf-8")
    return path
