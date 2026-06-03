"""QU100 paper-trade tracker (Phase 0+1).

Turns `setup_long` LLM theses into tracked $10k paper positions, fills them at
the next session's open, walks path-dependent stop-loss / target exits, and
persists a regenerable daily report. See
docs/DESIGN-qu100-llm-feedback-loop.md and
docs/TEST-SPEC-qu100-paper-tracker-p01.md.
"""
