"""Streamlit dashboard for the LLM thesis layer (PR4).

Three pure-Python modules ship here:

  * `data.py`    — read-only DB loaders + small dataclasses (NO Streamlit).
  * `actions.py` — write helpers (toggle signal, accept/reject insight,
                   test signal). Shared with the CLI accept/reject path.
  * `app.py`     — Streamlit multi-tab entrypoint. Run via:
                   `uv run streamlit run src/rainier/dashboard/app.py`.

Only `app.py` imports Streamlit. `data.py` + `actions.py` stay testable
without any Streamlit dependency, so unit tests can exercise the loaders
directly. Per the design + eng review, Streamlit page rendering is
covered by manual smoke only — we don't unit-test the UI layer.
"""

from __future__ import annotations

__all__ = [
    "data",
    "actions",
]
