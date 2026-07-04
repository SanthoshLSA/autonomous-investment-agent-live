"""
Report Generator Orchestrator.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from xhtml2pdf import pisa

from src.logger import get_logger
from src.reports.html_report import generate_html_report

logger = get_logger(__name__)


def generate_pdf_report(html_path: Path, pdf_path: Path) -> bool:
    """Converts the generated HTML report to a PDF document."""
    try:
        logger.info("Generating PDF report file...", path=str(pdf_path))
        with open(html_path, encoding="utf-8") as f:
            html_content = f.read()

        # Swap out complex CSS (unsupported by xhtml2pdf) for standard clean PDF print styling
        if "<style>" in html_content and "</style>" in html_content:
            parts_before = html_content.split("<style>")
            parts_after = parts_before[1].split("</style>")

            pdf_css = """
            body {
                font-family: Helvetica, Arial, sans-serif;
                color: #222222;
                background-color: #ffffff;
                line-height: 1.4;
                padding: 10px;
            }
            header {
                border-bottom: 2px solid #555555;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }
            header h1 {
                font-size: 24px;
                color: #111111;
            }
            header p {
                color: #666666;
                font-size: 11px;
            }
            .card {
                border: 1px solid #dddddd;
                border-radius: 6px;
                padding: 15px;
                margin-bottom: 20px;
                background-color: #fafafa;
            }
            .card-title {
                font-size: 16px;
                font-weight: bold;
                color: #7c3aed;
                border-bottom: 1px solid #eeeeee;
                padding-bottom: 5px;
                margin-bottom: 15px;
            }
            .stat-box {
                border: 1px solid #e5e7eb;
                padding: 10px;
                margin-bottom: 10px;
                background-color: #ffffff;
            }
            .stat-val {
                font-size: 18px;
                font-weight: bold;
                color: #7c3aed;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }
            th, td {
                text-align: left;
                padding: 8px;
                border-bottom: 1px solid #e5e7eb;
            }
            th {
                font-weight: bold;
                color: #4b5563;
                background-color: #f3f4f6;
            }
            .badge {
                font-weight: bold;
            }
            .badge-buy { color: #059669; }
            .badge-hold { color: #d97706; }
            .badge-sell { color: #dc2626; }
            .list-unstyled {
                list-style: none;
                padding-left: 0;
            }
            .list-unstyled li {
                margin-bottom: 5px;
            }
            """
            html_content = parts_before[0] + "<style>" + pdf_css + "</style>" + parts_after[1]

        # Clean inline CSS variables which cause xhtml2pdf to fail
        html_content = html_content.replace("var(--accent)", "#7c3aed")
        html_content = html_content.replace("var(--danger)", "#dc2626")
        html_content = html_content.replace("var(--success)", "#059669")
        html_content = html_content.replace("var(--warning)", "#d97706")

        with open(pdf_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)

        if pisa_status.err:
            logger.error("Failed to generate PDF report", err_code=pisa_status.err)
            return False

        logger.info("PDF report generation complete", path=str(pdf_path))
        return True
    except Exception as e:
        logger.error("Exception during PDF generation", error=str(e))
        return False


def generate_daily_report(state_data: dict[str, Any]) -> dict[str, Any]:
    """Orchestrates report file creation (HTML & PDF).

    Args:
        state_data: Analysis output dictionary.

    Returns:
        Dictionary mapping formats ('html', 'pdf') to their output file paths.
    """
    logger.info("Starting daily report generation...")
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Establish output directory structure
    out_dir = Path("reports/output") / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / "report.html"
    pdf_path = out_dir / "report.pdf"

    # Render HTML file
    generate_html_report(state_data, html_path)

    # Convert HTML to PDF
    pdf_success = generate_pdf_report(html_path, pdf_path)

    logger.info(
        "Daily report generation complete",
        html_path=str(html_path),
        pdf_path=str(pdf_path) if pdf_success else "None",
    )

    results = {
        "html": str(html_path.resolve()),
    }
    if pdf_success:
        results["pdf"] = str(pdf_path.resolve())
    return results
