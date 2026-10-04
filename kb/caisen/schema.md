# Output schemas

All outputs are UTF-8 JSON. Quote the book in Chinese verbatim; write analysis in English.
Every claim cites `pages` (PDF page numbers, as in the `pNNN.png` filenames).

## Reader notes — `runs/notes/<model>/<unit_id>.json`

```json
{
  "unit_id": "u01",
  "model": "opus-5-5",
  "pages_read": [17, 18],
  "summary": "2-4 sentences: what this unit teaches",
  "patterns": [
    {
      "name_zh": "W底", "name_en": "W bottom (double bottom)",
      "direction": "long|short|neutral",
      "structure": "how the shape is built (pivots, neckline, legs)",
      "entry": "exact entry trigger, e.g. close above neckline",
      "stop": "exact stop rule",
      "target": {"formula": "neckline + (neckline - low)", "worked_example": "18 + (19-14) = 23 ...", "pages": [17]},
      "confirmation": ["volume, time, retest, etc."],
      "invalidation": ["what kills the setup"],
      "timeframes": ["daily", "weekly", ...],
      "pages": [17, 18]
    }
  ],
  "rules": [
    {"id": "R1", "rule": "neckline breakout must not fall back below the neckline", "kind": "entry|stop|target|filter|risk|time|volume|psychology", "quote_zh": "…", "pages": [17]}
  ],
  "worked_examples": [
    {"instrument": "e.g. 台股加权指数", "timeframe": "daily", "period": "2008-03..2008-06", "pattern": "W底", "levels": {"neckline": 0, "target": 0}, "outcome": "hit target / failed / n.a.", "pages": [130]}
  ],
  "quantifiable": ["rules that can be coded as a detector, phrased as testable conditions"],
  "ambiguities": ["things the book leaves unclear or contradicts elsewhere"],
  "illegible": [{"page": 0, "what": "which part could not be read"}]
}
```

Units without trading patterns (preface, A-share commentary, afterword) may leave `patterns` empty
but must still fill `summary`, `rules` (principles, risk, psychology) and `worked_examples`.

## Challenge / critique — `runs/critiques/r<N>/<model>.json`

Round 1 critiques the other models' **notes**; rounds ≥ 2 critique the **synthesis draft**.

```json
{
  "round": 1,
  "model": "gpt-5-5",
  "target": "notes|synthesis",
  "findings": [
    {
      "id": "gpt-5-5-r1-001",
      "about": "opus-5-5/u03.json patterns[0].target  |  docs/caisen_methodology.md §3.2",
      "severity": "major|minor",
      "type": "factual_error|omission|contradiction|misread_number|overreach|unclear",
      "claim": "what the reviewed text says",
      "correction": "what the book actually says",
      "evidence_pages": [23],
      "quote_zh": "…"
    }
  ],
  "agree_with": ["finding ids from other models in the same round you endorse (optional, second pass)"]
}
```

`major` = would change a detector, target, stop, or thesis judgement. Everything else is `minor`.

## Synthesis — `docs/caisen_methodology.md` + `runs/synthesis/r<N>/changelog.json`

The changelog records how every finding of the previous round was handled:

```json
{"round": 2, "resolutions": [
  {"finding_id": "gpt-5-5-r1-001", "decision": "accepted|rejected|disputed", "reason": "…", "section": "§3.2"}
]}
```
