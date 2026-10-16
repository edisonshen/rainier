---
name: qu-ticker-analysis
description: On-demand QU100-style thesis for ONE ticker ("analyze NVDA"). Runs `rainier thesis collect` on the Mac mini (QU capitalflow + the daily thesis evidence + 200-bar chart), then the agent writes the thesis itself — no Claude API call, no DB or Discord writes.
---

# QU single-ticker analysis

Use when the user asks to analyze / give a thesis for a specific ticker.

## 1. Reach the Mac mini

The QU fetch needs the mini's logged-in Chrome (CDP on :9222) and the DB lives there.

```bash
TAILSCALE_TAG=tag:devin,tag:rainier-db scripts/devin-db-connect.sh
printf '%s\n' "$MAC_MINI_SSH_KEY" > ~/.ssh/mini_key && chmod 600 ~/.ssh/mini_key
MINI="ssh -i ~/.ssh/mini_key -o IdentitiesOnly=yes pinkbear@100.75.228.52"
```

If SSH times out: `sudo tailscale logout`, rerun the connect script, and check
`tailscale status | grep pinks` (the mini drops off for ~30 min when rebooting).

## 2. Collect the pack (read-only)

```bash
SYM=NVDA
$MINI "cd ~/Projects/rainier && /Users/pinkbear/.local/bin/uv run rainier thesis collect $SYM --out /tmp/thesis-packs/$SYM"
scp -r -i ~/.ssh/mini_key -o IdentitiesOnly=yes pinkbear@100.75.228.52:/tmp/thesis-packs/$SYM /tmp/
```

- Works for any ticker. Outside the QU100 "Long in" set the screener section shows
  `rank 0, "Not in QU100"` — that is expected, not an error.
- If it prints `QU capitalflow fetch failed`, the pack is still written without the QU
  section; say so in the answer. `--no-qu` skips the fetch on purpose.
- "not enough yfinance history" = new/illiquid listing; tell the user, don't retry.

## 3. Analyze

1. Read `/tmp/$SYM/prompt.md`. Its "System prompt" section is the exact instruction and
   JSON schema the daily Claude thesis uses — follow it (verdict, quality, confidence,
   levels, risks, watch items).
2. View `/tmp/$SYM/chart.png` (200 daily bars, pattern entry/SL/target lines).
3. `evidence.json` has the raw numbers (signals, QU capital-flow rows) if you need detail.

Reply in chat with: verdict + quality/confidence, entry/SL/target and R:R, the three
paragraphs, risks, watch items, and a one-line read of the QU daily/weekly flow trend.
Show the chart image in the reply.

## Never

- Call the Claude/Anthropic API or `rainier thesis daily|ticker` (those spend API money).
- Write to the DB, post to Discord, or edit the mini's `~/Projects/rainier` checkout.
