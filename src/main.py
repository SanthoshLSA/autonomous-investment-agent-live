"""
Main orchestration entry point for the Autonomous Investment Research Agent.

Provides a Command Line Interface (CLI) to trigger runs, execute historical backtests,
or start the scheduler daemon.
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Any

import click
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.agents.graph import (
    build_investment_graph,
    resume_with_approval,
    run_investment_analysis,
)
from src.config import get_api_keys, get_config
from src.logger import get_logger, setup_logging
from src.reports.generator import generate_daily_report

logger = get_logger(__name__)


def run_pipeline() -> dict[str, Any]:
    """Runs the full investment research pipeline end-to-end.

    Fetches, analyzes, recomends, checks approval (auto-approves in CLI mode),
    and generates output report HTML documents.

    Returns:
        Final state dictionary containing results.
    """
    logger.info("Starting scheduled investment analysis run...")
    config = get_config()
    api_keys = get_api_keys()

    graph = build_investment_graph(config, api_keys)
    watchlist = config.watchlist.all_tickers
    thread_id = f"cli-run-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    try:
        # 1. Run pipeline up to human approval gate
        logger.info("Invoking sequential LangGraph agents...")
        run_investment_analysis(graph, watchlist, thread_id)

        # 2. CLI execution auto-approves proposals for convenience
        logger.info("Auto-approving portfolio allocations in automated daemon run mode...")
        final_state = resume_with_approval(graph, thread_id, "approve")

        # 3. Generate HTML output reports
        reports = generate_daily_report(final_state)
        logger.info("Scheduled pipeline completed successfully", reports=reports)

        # 3.5. Update all Paper Trading accounts with daily closing valuations
        try:
            from src.portfolio.valuation_engine import run_all_valuations

            run_all_valuations()
        except Exception:
            logger.exception("Failed to run paper trading valuations updates")

        return final_state

    except Exception as e:
        logger.exception("Critical pipeline runtime error")
        raise e


@click.group()
def cli() -> None:
    """Autonomous Investment Research Agent CLI."""
    # Ensure logs directory and logging configurations are loaded
    config = get_config()
    setup_logging(
        level=config.logging.level,
        log_format=config.logging.format,
        file_path=config.logging.file_path,
    )


@cli.command("run")
def run_now() -> None:
    """Triggers the full scanner, advisor, and report generator immediately."""
    logger.info("Executing immediate scan run...")
    try:
        run_pipeline()
        click.echo("[SUCCESS] Scan completed successfully. Check reports/output/ directory.")
    except Exception as e:
        click.echo(f"[ERROR] Error running scan: {str(e)}", err=True)
        sys.exit(1)


@cli.command("daemon")
def start_daemon() -> None:
    """Launches the scheduler daemon running scans daily at 9:00 AM IST."""
    config = get_config()
    if not config.schedule.enabled:
        click.echo("[WARNING] Scheduling is disabled in config.yaml.")
        return

    run_time = config.schedule.run_time  # format '09:00'
    hour, minute = map(int, run_time.split(":"))

    scheduler = BlockingScheduler(timezone=config.schedule.timezone)

    # Configure trigger parameters (skipping weekends if configured)
    day_of_week = "mon-fri" if config.schedule.weekend_skip else "*"
    trigger = CronTrigger(hour=hour, minute=minute, day_of_week=day_of_week)

    scheduler.add_job(
        run_pipeline,
        trigger=trigger,
        id="daily_investment_scan",
        name="Daily Watchlist Portfolio Research Scan",
    )

    logger.info(
        "Scheduler daemon started successfully",
        time=run_time,
        timezone=config.schedule.timezone,
        weekends_skipped=config.schedule.weekend_skip,
    )
    click.echo(f"[INFO] Scheduler running daily at {run_time} ({config.schedule.timezone})")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler daemon shut down gracefully.")
        click.echo("Graceful exit.")


if __name__ == "__main__":
    cli()
