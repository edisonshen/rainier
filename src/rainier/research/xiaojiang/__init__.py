"""Xiaojiang S/R line system, rebuilt from her trading notes (research only).

Rules taken from the doc:
- Lines are horizontal S/R from 5m/15m up to 1D; big-timeframe ("thick") lines matter most and
  multi-timeframe confluence makes a line stronger. Boxes (ranges) are lines too.
- Contrarian, left side: resting limits at the line, buy support, short resistance. Right side
  variant: enter after price touches the line and rejects (e.g. "on the 3rd 5m candle").
- The day's range is the 2-3 lines above and below price.
- Fixed stop beyond the line (ES 6 pts; double for MES -> 12).
- Take profit at the next line, scale out line by line, or 1:1 / 1:2 for beginners.
- First touch beats later touches; many touches mean the line is likely to break.
- If the day's bias is down, sell the upper lines; if up, buy the lower lines; range: both.

Implementation choices (not from the doc) are flagged in the parameter docs.
"""
