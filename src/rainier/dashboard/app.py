"""Streamlit dashboard entrypoint — `uv run streamlit run src/rainier/dashboard/app.py`.

Four tabs, all data sourced from `dashboard.data` (read-only) and all
writes routed through `dashboard.actions` so the CLI and the UI share
the same code path. The UI layer is deliberately thin — page logic
should not contain business rules. Add new logic to `data.py` /
`actions.py` and unit-test it; render here.

Smoke-tested manually only. Per the eng review (design doc Section G),
Streamlit page rendering is excluded from unit tests; the data + action
helpers are.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from rainier.dashboard import actions, data
from rainier.llm_thesis.signals import REGISTRY

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Rainier LLM Thesis",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Tab 1 — Signals (config + live status)
# ---------------------------------------------------------------------------


def _render_signals_tab() -> None:
    st.header("Signals")
    st.caption(
        "Toggle signals on/off and inspect their last-7d hit count. "
        "Edits write back to config/settings.yaml via ruamel "
        "(comments preserved)."
    )

    signal_rows = data.load_signal_status()
    if not signal_rows:
        st.info("No signals registered.")
        return

    df = pd.DataFrame(
        [
            {
                "name": r.name,
                "enabled": r.enabled,
                "version": r.version,
                "weight": r.weight,
                "cost_estimate_ms": r.cost_estimate_ms,
                "last_7d_hit_count": r.last_7d_hit_count,
            }
            for r in signal_rows
        ]
    )

    edited = st.data_editor(
        df,
        column_config={
            "name": st.column_config.TextColumn("name", disabled=True),
            "enabled": st.column_config.CheckboxColumn("enabled"),
            "version": st.column_config.TextColumn("version", disabled=True),
            "weight": st.column_config.NumberColumn(
                "weight", disabled=True, format="%.2f"
            ),
            "cost_estimate_ms": st.column_config.NumberColumn(
                "cost_estimate_ms", disabled=True
            ),
            "last_7d_hit_count": st.column_config.NumberColumn(
                "last_7d_hits", disabled=True
            ),
        },
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        key="signals_editor",
    )

    # Diff against the original frame; flip whatever changed.
    changed = []
    for original, new in zip(
        df.itertuples(index=False), edited.itertuples(index=False)
    ):
        if bool(original.enabled) != bool(new.enabled):
            changed.append((original.name, bool(new.enabled)))
    if changed:
        for name, enabled in changed:
            try:
                actions.toggle_signal(name, enabled)
                st.toast(
                    f"Set {name} enabled={enabled}", icon="OK"
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"toggle_signal({name}) failed: {exc}")
        st.rerun()

    st.divider()
    with st.expander("Test a signal"):
        col1, col2 = st.columns(2)
        with col1:
            test_signal_name = st.selectbox(
                "Signal",
                options=sorted(REGISTRY.keys()),
                key="test_signal_name",
            )
        with col2:
            test_symbol = st.text_input(
                "Symbol", value="NVDA", key="test_symbol"
            )
        if st.button("Run test", key="run_test_signal"):
            try:
                with st.spinner(f"Running {test_signal_name} on {test_symbol}..."):
                    result = actions.test_signal(test_signal_name, test_symbol)
                st.success(f"{result['signal']} on {result['symbol']}")
                st.code(result.get("render_for_prompt") or "(no output)")
                st.json(result.get("value") or {})
            except Exception as exc:  # noqa: BLE001
                st.error(f"test_signal failed: {exc}")


# ---------------------------------------------------------------------------
# Tab 2 — Performance
# ---------------------------------------------------------------------------


_FALSIFICATION_LIFT_THRESHOLD = 0.001
_FALSIFICATION_PVALUE_THRESHOLD = 0.20


def _render_performance_tab() -> None:
    st.header("Performance")
    st.caption(
        "Per-signal contribution (Mann-Whitney U used vs absent forward "
        "returns) and per-verdict hit rate. Red rows indicate falsified "
        "signals — flat lift AND no statistical signal over the window."
    )

    window = st.selectbox(
        "Rolling window (days)",
        options=[7, 14, 30, 60],
        index=2,
        key="perf_window",
    )

    contribs = data.load_signal_contribution(days=int(window))
    if contribs:
        contrib_df = pd.DataFrame(
            [
                {
                    "name": c.name,
                    "n_used": c.n_used,
                    "n_absent": c.n_absent,
                    "mean_used": c.mean_used,
                    "mean_absent": c.mean_absent,
                    "lift": c.lift,
                    "p_value": c.p_value,
                    "falsified": _is_falsified(c.lift, c.p_value),
                }
                for c in contribs
            ]
        )
        st.subheader("Per-signal contribution")
        # Highlight falsified rows with a red background; keep the rest
        # default-themed.
        st.dataframe(
            contrib_df.style.apply(
                lambda row: [
                    "background-color: #ffe1e1"
                    if bool(row.get("falsified"))
                    else ""
                ]
                * len(row),
                axis=1,
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("Lift over time")
        # Each signal gets its own line — eng review wanted a rolling
        # picture, not just a current snapshot.
        lift_df = contrib_df.set_index("name")[["lift"]]
        st.bar_chart(lift_df, use_container_width=True)
    else:
        st.info("No ThesisEvaluation rows in the selected window yet.")

    # Per-signal scatterplot — value vs forward return.
    st.subheader("Signal value vs forward return")
    for name in sorted(REGISTRY.keys()):
        with st.expander(f"{name} — scatter"):
            df = data.load_signal_value_series(name, days=int(window))
            if df.empty:
                st.caption(
                    f"No value/outcome pairs for `{name}` in the window."
                )
                continue
            chart_df = df.rename(
                columns={"signal_value": "x", "forward_return_5d": "y"}
            )
            st.scatter_chart(chart_df, x="x", y="y", use_container_width=True)
            st.caption(f"n = {len(df)}")

    # Per-verdict hit rate.
    st.subheader("Verdict hit rate")
    by_verdict = data.load_verdict_hit_rate(days=int(window))
    rows: list[dict[str, Any]] = []
    for verdict, hr_list in by_verdict.items():
        for hr in hr_list:
            rows.append(
                {
                    "verdict": verdict,
                    "horizon": hr.horizon,
                    "n": hr.n,
                    "win_rate": hr.win_rate,
                    "avg_return_pct": hr.avg_return_pct,
                }
            )
    if rows:
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, use_container_width=True
        )
    else:
        st.info("No verdict outcomes available yet.")


def _is_falsified(lift: float, p_value: float | None) -> bool:
    """Red-flag rule (design Section G Tab 2): p>threshold AND lift<=threshold.

    None p-value means too few samples — not falsified, just unproven.
    """
    if p_value is None:
        return False
    return (
        p_value > _FALSIFICATION_PVALUE_THRESHOLD
        and lift <= _FALSIFICATION_LIFT_THRESHOLD
    )


# ---------------------------------------------------------------------------
# Tab 3 — Recent theses
# ---------------------------------------------------------------------------


def _render_theses_tab() -> None:
    st.header("Recent theses")
    st.caption(
        "Recent LLM-augmented picks with forward returns at 1d/5d/10d "
        "(NULL until the daily eval job has run for that horizon)."
    )

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        since = st.date_input(
            "Since",
            value=date.today() - timedelta(days=30),
            key="theses_since",
        )
    with col2:
        until = st.date_input(
            "Until",
            value=date.today(),
            key="theses_until",
        )
    with col3:
        verdicts = st.multiselect(
            "Verdict",
            options=["setup_long", "watch", "no_setup"],
            default=[],
            key="theses_verdict",
        )
    with col4:
        symbol = st.text_input("Symbol", value="", key="theses_symbol")

    rows = data.load_recent_theses(
        limit=50,
        verdicts=verdicts or None,
        symbol=symbol or None,
        since=since,
        until=until,
    )
    if not rows:
        st.info("No theses match the filter.")
        return

    df = pd.DataFrame(
        [
            {
                "thesis_id": r.thesis_id,
                "scan_date": r.scan_date,
                "session": r.session_name,
                "symbol": r.symbol,
                "verdict": r.verdict,
                "confidence": r.llm_confidence,
                "signals": ", ".join(r.signals_used) if r.signals_used else "",
                "return_1d": r.return_1d,
                "return_5d": r.return_5d,
                "return_10d": r.return_10d,
            }
            for r in rows
        ]
    )
    st.dataframe(df, hide_index=True, use_container_width=True)

    selected_id = st.selectbox(
        "Inspect thesis (id)",
        options=[None, *df["thesis_id"].tolist()],
        format_func=lambda x: "—" if x is None else str(x),
        key="theses_inspect",
    )
    if selected_id is None:
        return
    selected = next((r for r in rows if r.thesis_id == selected_id), None)
    if selected is None:
        return
    st.markdown(f"### {selected.symbol} — {selected.verdict}")
    if selected.structured_output:
        st.json(selected.structured_output)
    chart_bytes = data.load_thesis_chart(selected.thesis_id)
    if chart_bytes:
        st.image(chart_bytes, caption=f"Chart for {selected.symbol}")
    else:
        st.caption("No chart on disk for this thesis.")


# ---------------------------------------------------------------------------
# Tab 4 — Insights
# ---------------------------------------------------------------------------


def _severity_color(severity: str) -> str:
    # Matches the Discord ordering — readable on both light + dark themes.
    return {
        "critical": ":red[CRITICAL]",
        "warn": ":orange[WARN]",
        "info": ":blue[INFO]",
    }.get(severity, severity)


def _render_insights_tab() -> None:
    st.header("Research insights")
    st.caption(
        "Pending insights (top) and audit trail (bottom). Accept applies "
        "the structured action via ACTION_EXECUTORS; reject stores the "
        "reason and leaves settings.yaml untouched."
    )

    pending = data.load_pending_insights()
    st.subheader(f"Pending ({len(pending)})")
    if not pending:
        st.success("No pending insights.")
    for r in pending:
        with st.expander(
            f"#{r.insight_id} — {r.kind} — {r.subject} "
            f"({_severity_color(r.severity)})"
        ):
            st.write(r.rationale or "_(no rationale)_")
            action = r.action or {}
            st.markdown(
                f"**Action:** `{action.get('kind', '?')}` "
                f"-> `{action.get('target', '?')}`"
            )
            st.markdown(f"**Recurrence count:** {r.recurrence_count}")
            with st.expander("Evidence (raw JSON)"):
                st.json(r.evidence or {})

            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                if st.button("Accept", key=f"accept_{r.insight_id}"):
                    try:
                        result = actions.accept_insight(r.insight_id)
                        st.toast(
                            f"Accepted #{r.insight_id} -> {result['action_kind']}",
                            icon="OK",
                        )
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Accept failed: {exc}")
            with col2:
                reason = st.text_input(
                    "Reject reason",
                    key=f"reject_reason_{r.insight_id}",
                    placeholder="Why are we dismissing this?",
                )
            with col3:
                if st.button("Reject", key=f"reject_{r.insight_id}"):
                    if not reason:
                        st.warning("Provide a reason before rejecting.")
                    else:
                        try:
                            actions.reject_insight(r.insight_id, reason)
                            st.toast(
                                f"Rejected #{r.insight_id}", icon="OK"
                            )
                            st.rerun()
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Reject failed: {exc}")

    st.divider()
    with st.expander("History (accepted / rejected / auto-applied)"):
        statuses = st.multiselect(
            "Status filter",
            options=["accepted", "rejected", "auto_applied", "stale"],
            default=["accepted", "rejected"],
            key="hist_status",
        )
        history = data.load_insight_history(status=statuses or None)
        if not history:
            st.info("No history rows match the filter.")
        else:
            hist_df = pd.DataFrame(
                [
                    {
                        "id": r.insight_id,
                        "kind": r.kind,
                        "subject": r.subject,
                        "severity": r.severity,
                        "status": r.status,
                        "decided_at": r.decided_at,
                        "decided_by": r.decided_by,
                    }
                    for r in history
                ]
            )
            st.dataframe(
                hist_df, hide_index=True, use_container_width=True
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("Rainier — LLM thesis dashboard")
    tab_signals, tab_perf, tab_theses, tab_insights = st.tabs(
        ["Signals", "Performance", "Recent theses", "Insights"]
    )
    with tab_signals:
        _render_signals_tab()
    with tab_perf:
        _render_performance_tab()
    with tab_theses:
        _render_theses_tab()
    with tab_insights:
        _render_insights_tab()


# Streamlit imports the module and runs it top-to-top, so the entrypoint
# call must be at module scope.
main()
