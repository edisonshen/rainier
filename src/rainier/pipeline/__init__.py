"""Shared post-scrape pipeline glue.

The ``pipeline`` package hosts orchestration helpers that compose multiple
domain modules (analysis, llm_thesis, alerts) into a single callable.
Domain modules themselves stay free of cross-imports — this layer is the
composition point.

Currently exposes:

* :func:`rainier.pipeline.post_scrape.run_post_scrape_pipeline` — used by
  both ``rainier scrape qu`` (cron CLI path) and the long-running
  ``rainier run`` scheduler service.
"""

from rainier.pipeline.post_scrape import run_post_scrape_pipeline

__all__ = ["run_post_scrape_pipeline"]
