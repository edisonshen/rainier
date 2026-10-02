# DESIGN — Recursive strategy improvement for Rainier

**Status:** research + plan, for operator review. Nothing implemented.
**Question this answers:** how do we make Rainier improve its own trading strategy on a loop, without the loop quietly overfitting itself into a mirage?
**Scope:** the QU100 stock screener + LLM-thesis + paper-trade stack (the futures/pin-bar side and the TQQQ/ETF side ride along on the same harness).

---

## 0. TL;DR

Rainier already has **four half-built improvement loops**. They are not the bottleneck.

The bottleneck is that **the thing they optimize is not measured correctly yet**, so making the loop faster or more autonomous would amplify noise, not edge. Three concrete measurement defects (§2) are visible in the repo's own committed evidence.

So the plan is ordered: **fix the ruler → build the promotion gate → then, and only then, turn on recursion.**

| Phase | What it buys | Sessions |
|---|---|--:|
| P0 — Fix the ruler | Outcome metric that measures *edge*, not market beta | 2–3 |
| P1 — Promotion gate + trial ledger | A single door every change must walk through; overfitting becomes measurable | 2–3 |
| P2 — Reward registry + L3 evaluator (Slice 1) | Turn "a strategy variant" into a scored, reproducible object | 2–3 |
| P3 — LLM proposer + MAP-Elites archive (Slice 2) | Actual recursion: the system proposes its own variants | 3–4 |
| P4 — Bandit allocation over live paper | Recursion touches capital, safely | 2 |
| P5 — Meta-loop (evolve the proposer, widen auto-apply) | Self-improvement of the improver | 2 |

Total ≈ **13–18 sessions**, phase-gated — each phase is independently useful and P0/P1 are worth doing even if you stop there.

The one public account of an industrial version of this loop running (§3, ref 15) reinforces the ordering: it succeeded on tasks chosen for *tight feedback, clear metrics, low variance, and hardenable evaluators* — the four properties a QU100 backtest lacks and that P0/P1 exist to manufacture.

---

## 1. What already exists (audit)

Rainier is much further along than "we should build a feedback loop". Four loops are live or stubbed:

### Loop A — weekly auto-research → config actions *(live, human-gated)*

`llm_thesis/research.py` (1590 LOC). Friday 09:00 PT. Eight checks
(`signal_underperform`, `signal_overperform`, `verdict_drift`, `calibration_off`,
`new_pattern_discovered`, `prompt_regression`, `time_stop_discovered`,
`paper_lessons`) emit `ResearchInsight` rows; the operator accepts/rejects in the
Streamlit **Insights** tab; accept dispatches through `ACTION_EXECUTORS` into
`config/settings.yaml` via round-trip ruamel.

**Ceiling:** the action space is *five* executors — `disable_signal`,
`bump_prompt_version`, `raise_signal_weight`, `lower_signal_weight`,
`set_learned_time_stop_days`. That's the entire set of changes this loop can
make to the strategy. And `ThesisSignalConfig.weight` is still inert
(`DESIGN-qu100-llm-feedback-loop.md` D7b), so two of the five are placebos today.

### Loop B — paper trades → prompt context *(live)*

`paper/` (≈5k LOC): `positions` → `exit.evaluate_exit` → `calibration` (headline
= unbiased `ThesisEvaluation` fixed-horizon stats, plus realized *and* MTM paper
stats) → injected into `build_user_message()`; `reflection.py` asks the LLM to
post-mortem each closed trade and feeds the last K back into the prompt.

This is the **fastest and cheapest** channel — it changes model behavior on the
next scan, with no config mutation and no deploy. It is also the loop most at
risk of teaching the model a survivorship-skewed lesson, which D7a/(a2) already
anticipates.

### Loop C — champion/challenger for the screener *(loader only)*

`core/champion.py` + `config/model/champion.yaml` (v1, seeded byte-identical to
`settings.yaml`, `score: null`, no promotions yet) + `history/` +
`registry.parquet`.

**Ceiling:** the *loading* and *versioning* half exists. The half that
generates challengers, scores them out-of-sample, and decides promotion does not.
`champion.yaml` has never been promoted past the behavior-preserving seed.

### Loop D — the actual recursion engine *(Slice 0 stubs)*

`research/` is scaffolded for an AlphaEvolve-shaped system and explicitly labels
what's missing:

- `research/rewards/__init__.py` — `REGISTRY: dict = {}`, "seven pre-registered reward functions land in Slice 1"
- `research/evaluator/trade_simulator.py` — `SLICE_THIS_LANDS_IN = 1`
- `research/archive/__init__.py` — "L4 MAP-Elites archive — Slice 2"
- `research/bandit/__init__.py` — "L5 Thompson bandit — Slice 2"
- Shipped: `providers/` (anthropic/deepseek/openai-compatible), `cost_pilot.py`, `survivorship.py`, `output_schema.py`, `schemas.py:ResearchThesis` with hallucination guards.

**This is the recursive loop you're asking about, and it is ~15% built.** The
plan below is essentially "finish Loop D, but fix the evaluator's ruler first,
because Loop D is a machine for exploiting whatever the ruler rewards."

### Supporting assets worth reusing

`backtest/walk_forward.py`, `ml/compare.py:run_walkforward_compare`,
`paper/pattern_replay.py` (parity-tested live-detector replay),
`paper/pattern_audit.py` (231k-emission forward-return corpus),
`paper/sweep.py` (weekly missed-winner attribution),
`research/survivorship.py`, `research/cost_pilot.py` (per-call $ accounting).

---

## 2. Three defects that must be fixed *before* recursion

These are not hypotheses — each is visible in committed artifacts.

### D-1 — The outcome metric measures market beta, not pattern edge 🔴

`docs/REPORT-qu100-pattern-hit-rate.md` is the evidence base for the pattern
weights that carry **65% of the screener score**. It reports *raw* forward
returns. Read the bear-regime rows:

| pattern | direction | regime | H=10 | mean fwd | dir-correct |
|---|---|---|--:|--:|--:|
| `false_breakdown` | bullish | bear | 2282 | **+6.8%** | 82.0% |
| `hs_bottom` | bullish | bear | 331 | **+5.5%** | 77.3% |
| `m_top` | **bearish** | bear | 2199 | **+4.7%** | 13.8% |
| `hs_top` | **bearish** | bear | 625 | **+5.3%** | 14.3% |
| `bear_flag` | **bearish** | bear | 988 | **+6.6%** | 15.7% |

Everything tagged in a "bear" regime returned **+5% to +7% over 10 days,
regardless of what the pattern said**. Bullish patterns look brilliant (82%
dir-correct); their bearish mirrors look catastrophically inverted (14%). Both
readings are the same fact: *the sample's bear-regime days were followed by a
market-wide rebound.* The pattern contributed approximately nothing.

Now look at bull/unknown regimes — 46–56% dir-correct on every single pattern,
i.e. **coin flips**, with mean forward returns of ±0.5%.

**Implication:** a recursive optimizer pointed at this metric will not discover
better patterns. It will discover *regime-timing dressed as pattern selection*,
and it will do so with high confidence because the numbers are enormous.

**Fix:** the corpus must carry **excess return vs a benchmark** — minimum SPY,
better SPY + the name's sector ETF (a 2-factor residual) — as the primary column,
with raw return demoted to context. Everything downstream (weights, rewards,
promotion) scores on excess.

### D-2 — The regime label is undefined for most of the corpus 🟠

The same report: `unknown` (fewer than 200 SPY bars at/before the as-of day)
dominates. For `w_bottom`: 29,103 unknown vs 8,540 bull vs 1,067 bear. Over a
365-day window ending 2026-06-16, ~200-bar SPY history should exist for
essentially every day — so this is a **data gap in the SPY series**, not a
genuine cold start. Two-thirds of the evidence base is regime-blind, and the
`bear` cell that drives every "significant" finding is the smallest.

**Fix:** backfill SPY (`market.benchmark_ohlcv` already exists and has its own
cron since `benchmark-ohlcv-spy-stale`) and re-derive; then replace the binary
SPY-vs-200SMA label with a small continuous regime vector — realized vol
percentile, breadth (you compute it already in `market_breadth/`), Fear & Greed
(`data/fear_greed.py`, ingest built but cron `enabled: false`), term structure.
Binary regime + tiny bear cell is exactly the shape that produces confident
nonsense.

### D-3 — Nothing anywhere controls for multiple testing 🔴

`grep -riE "purge|embargo|deflat|pbo|bonferroni|benjamini" src/` returns **zero**
hits for all of them. What exists: raw uncorrected Mann-Whitney p-values in
`research.py` and `ml/feature_selector.py`, and walk-forward splits in
`backtest/walk_forward.py` with **no purge/embargo** between train and test (so
overlapping multi-day forward-return labels leak across the boundary).

Meanwhile the system already runs `--sweep`, `--variations`, `--patterns`,
`tqqq_sma_sweep`, `paper/sweep`, and picks the in-sample max. The count of
implicit trials is already in the thousands and is **not recorded anywhere**.

A recursive loop multiplies trial count by 100–1000×. Without a trial ledger,
the Deflated Sharpe Ratio and the Probability of Backtest Overfitting are not
even computable — you don't know `N`.

**Fix:** P1 below. This is the single highest-leverage piece of the whole plan.

### Two smaller ones

- **D-4 — Asymmetric realization.** D6's "no hardcoded max-hold" baseline lets
  losers realize (stop-loss) while winners stay open indefinitely, so any
  realized-only statistic is biased. `calibration.py` already handles this for
  the prompt; the *reward function* must too — recursion needs a fixed
  time-cap in the evaluator even while live paper trading stays uncapped.
- **D-5 — No null baseline.** No check anywhere answers "would a coin flip with
  the same trade frequency and the same regime exposure have scored this well?"
  Every promotion should clear a permutation/shift-the-signal control.

---

## 3. Target architecture — four loops, four clock speeds

The literature that matters here (AlphaEvolve-lineage applied to trading:
*MadEvolve* 2605.23007, *QuantEvolve*, *AlgoEvolve* 2606.26173, *EvoQuant*
2607.12455, *QuantaAlpha* 2602.07085, *AEL* 2604.21725) converges on the same
skeleton, and it maps almost one-to-one onto Rainier's existing `research/`
package layout:

```
 L3  meta        evolve the PROPOSER (prompt, reward set)   quarterly   human-gated
      │                                                                  ▲
 L2  allocation  Thompson bandit over archive elites        daily        capital
      │          → which elite gets the next paper sleeve                ▲
 L1  structure   LLM proposes strategy VARIANTS             weekly       archive
      │          → MAP-Elites archive keyed by behavior                  ▲
 L0  parameters  numeric optimizer fits each variant        per-variant  sealed
                 → purged walk-forward, cost-inclusive                   holdout
```

Two design commitments, both taken straight from what the literature found
necessary and from Rainier's own D-1/D-3:

**(a) Quality-diversity, not a single champion.** A MAP-Elites archive keyed on
*behavior descriptors* — trade frequency, median hold length, regime exposure
(bull/bear/vol bucket), pattern family, gross exposure — keeps a grid of
"best strategy in each behavioral cell" rather than one global max. This is the
main structural defense against the loop collapsing onto one overfit peak, and
it gives the L2 bandit a genuinely diverse arm set. `research/archive/` is
already reserved for exactly this.

**(b) One door.** Every promotion — a pattern weight change from Loop A, a
champion.yaml bump from Loop C, an evolved variant from Loop D — walks through
*the same* gate (§4, P1). No side doors. A recursive system with two promotion
paths has one promotion path plus a leak.

**Division of labor (the LATSS/MadEvolve split, worth copying):** the LLM
proposes *structure* and never picks numeric values or sees raw market data; the
numeric optimizer fits parameters; the harness judges. This keeps LLM
hallucination out of the number line and makes each proposal cheap to validate.

### Calibration against a working automated-research loop

Recursive's *First Steps Toward Automated AI Research* (ref 15) is the closest
public account of an industrial version of this loop actually running, so it is
worth checking the plan against it. Their loop is the same shape as §3's —
propose → implement → run → **validate** → let the result choose the next
experiment, many threads, retained context, merged branches. Four of their
operational choices change decisions in §4, and one disanalogy dominates
everything.

**They re-ran the incumbent before comparing to it.** Their headline number for
the prior state of the art was not the number the prior holder published — they
stripped its reward hacks, re-evaluated it on 10 random seeds, and quoted *that*.
Rainier's incumbent gets no more trust: the current champion's numbers were
produced by the pre-P0 harness on the raw-return corpus D-1 shows is scoring
beta. → P1.6, P2 acceptance.

**The incumbent was partly cheating.** That prior best came from a public
collaborative effort — dozens of humans and hundreds of agents — and still
contained reward hacks that survived until someone specifically looked. A
baseline is not clean because many eyes produced it. → §5.

**Two runs from different seeds converged on different, equally competitive
solutions.** They read this as evidence of real search rather than memorized
recall, which it is. It is *also* evidence that the top of the ranking is
noise-limited: when several structurally different candidates tie, picking the
argmax is picking noise. That is the case for the archive (§3a) rather than a
leaderboard, and financial data is far noisier than fixed-seed LM training. → P3.

**The weak starting point beat the community's best.** Starting from a vanilla
Transformer + AdamW, their system still passed the collaboratively-optimized
solution. Anchoring search on the incumbent is not obviously optimal; seeding
some archive cells from a deliberately naive baseline costs little and guards
against inheriting the champion's mistakes. → P3.

**The disanalogy that matters more than any of the above.** They chose their
benchmarks for *tight feedback loops, clear metrics, relatively low variance,
and evaluators that can be hardened against reward hacks* — and even with all
four they still ran 10 seeds and screened outputs by hand. A QU100 backtest has
none of the four: feedback takes weeks, the metric is contested (D-1), variance
swamps the effect size, and the evaluator leaks (D-3). Pointed at Rainier today,
the same system would mostly discover flaws in the evaluator. This is not an
argument against the approach — it is the argument for P0/P1, which exist
precisely to manufacture those four properties before any search runs.

One closing note in their own words: they flag that they may have missed reward
hacks in the kernel results "where we are not specialists." Nobody will reliably
eyeball a subtle look-ahead leak in an LLM-written strategy either, which is why
the P3 proposal surface starts at typed config mutations and not free Python.

---

## 4. Phased plan

Each phase: what, where it lands, acceptance criterion, estimate. Phases are
sequential; each ships independently.

### P0 — Fix the ruler (2–3 sessions)

1. **Excess-return corpus.** Add benchmark-relative columns to
   `paper/pattern_audit.py`'s corpus: `fwd_excess_spy_{5,10,20}` and
   `fwd_excess_sector_{5,10,20}`. Regenerate `docs/REPORT-qu100-pattern-hit-rate.md`
   with excess as the headline and raw demoted. Source: `market.benchmark_ohlcv`
   (SPY, already cron-refreshed) + sector ETF mapping.
2. **Repair the regime label.** Backfill SPY so `unknown` collapses to ~0;
   re-derive. Add the continuous regime vector (vol percentile, breadth, F&G) as
   corpus columns; arm the `fear-greed-daily` cron (currently
   `enabled: false` by design — needs operator approval to arm).
3. **Publish the delta.** A short `docs/REPORT-*-excess.md` diffing old vs new
   conclusions per pattern. Expect several patterns' apparent edge to evaporate;
   that is the deliverable, not a failure.

*Acceptance:* every per-(pattern, regime, horizon) cell reports excess return
with an n and a CI; `unknown` < 5% of emissions; the report states which of the
12 pattern weights the new evidence contradicts.

### P1 — The promotion gate + trial ledger (2–3 sessions)

The single most important phase. New module `research/gate.py`.

1. **Trial ledger** (`research_trial` table): every scored candidate ever
   evaluated — config hash, code hash, reward name, data window, score,
   parent id, timestamp. Append-only. This is what makes `N` knowable.
2. **Purged + embargoed walk-forward.** Extend `backtest/walk_forward.py` with a
   purge window ≥ the label horizon and an embargo after each test fold, so
   overlapping forward-return labels can't leak. Retrofit `ml/compare.py`.
3. **Sealed holdout.** Carve the most recent N months out of every research path
   and refuse to read it except at promotion, once, logged. Enforced in code
   (the loader raises), not by convention.
4. **Statistical gate**, computed against the ledger's trial count:
   - Deflated Sharpe Ratio (Bailey & López de Prado) — corrects for both
     selection bias under `N` trials and non-normal returns.
   - PBO via CSCV — "does my in-sample winner beat the median out-of-sample?"
     Promote only at PBO < 0.5, realistically < 0.3.
   - Null control: shift-the-signal / block-permutation test on the *active*
     (excess) return, so a strategy that merely re-times its own positions scores 0.
   - Cost stress: re-score at 2× assumed slippage/commission; edge must survive.
5. **Wire the existing loops through it.** `apply_action` and any
   `champion.yaml` promotion call `gate.check()` and record the verdict.
6. **Re-baseline the incumbent.** The champion's recorded numbers are not a
   valid comparison point — they predate P0's ruler. Before the gate can reject
   anything, re-score the current champion *through the gate itself* (same
   corpus, same purge/embargo, same cost stress) and write that as ledger entry
   zero. Audit it for the failure modes the gate is built to catch — reliance on
   `unknown`-regime rows, on the missed-winner sweep, on the un-embargoed folds —
   and record what it depends on. Ref 15 did exactly this before quoting a prior
   state of the art, and found reward hacks in a baseline that hundreds of
   agents had already picked over.

*Acceptance:* `champion.yaml` v1 → v2 cannot be written without a ledger entry
and a passing gate verdict; a deliberately overfit toy strategy is rejected by
the gate in a test; ledger entry zero is the re-scored champion, not its
historical self-report.

### P2 — Reward registry + L3 evaluator (Slice 1) (2–3 sessions)

Fills the stubs the repo already declares.

- **`research/rewards/`** — the seven pre-registered rewards (D-007). Pre-register
  them *before* running the search so reward-shopping is impossible. Suggested
  set: excess-return R-multiple expectancy, risk-adjusted (Sharpe on excess),
  Calmar on excess, hit-rate × payoff, all-call R (includes `no_setup` as a
  scored abstention), cost-adjusted expectancy, and a regime-robustness score
  (worst-regime performance, not average).
- **`research/evaluator/trade_simulator.py`** — entry-TTL gating, pessimistic
  fill (worst observed slippage in the entry bar), **mandatory time-cap exit**
  (D-4). Reuse `paper/exit.py:evaluate_exit`'s rules so research and live paper
  can't diverge.
- **`research/evaluator/builder.py`** — replace the dry-run stub with the real
  EvidencePack (OHLCV + signals + chart), sharing `paper/pattern_replay.py`'s
  parity-tested replay so a research score and a live emission mean the same thing.

- **Variance budget.** A single backtest path is one sample, and the evaluator
  must report dispersion, not just a point score: score every candidate over
  resampled paths (CSCV folds, bootstrapped entry-date jitter) and carry
  `(score, spread, n_paths)` through to the gate. Ref 15 used 10 seeds on a
  benchmark they describe as *low* variance; Rainier's is not low variance, so a
  point estimate is not a score. Candidates whose spread swamps their edge are
  rejected at the gate as unresolved, distinct from rejected as unprofitable.

*Acceptance:* re-scoring today's live champion through the L3 evaluator
reproduces the paper book's realized stats within tolerance (a parity test). If
it doesn't, the evaluator is wrong and nothing downstream is trustworthy.

### P3 — LLM proposer + MAP-Elites archive (Slice 2) (3–4 sessions)

Now recursion is safe to switch on.

- **Proposal surface, staged narrow → wide.** Stage 1: structured config
  mutations (weights, thresholds, layer mix, filters) — a bounded, typed,
  auto-validatable space. Stage 2: new *rule predicates* from a small DSL.
  Stage 3 (only if 1–2 pay off): free Python `SignalEmitter` subclasses, sandboxed.
  Do **not** start at stage 3; `AlgoEvolve`/`EvoQuant` both report hallucinated
  edits and strategy drift as the dominant failure mode there.
- **`research/archive/`** — MAP-Elites grid over the behavior descriptors in §3(a);
  each cell keeps its elite plus lineage (parent id → auditable ancestry, the
  fix `QuantaAlpha` proposes for untraceable evolution).
- **Hypothesis-first mutation.** The proposer must emit a stated *economic
  mechanism* alongside each variant, and that hypothesis is stored and later
  scored against outcome. Mutations without a mechanism are the ones that
  overfit; making the hypothesis a required, judged field is the cheapest
  available regularizer.
- **Seed from two starting points, not one.** Initialize part of the archive
  from the champion and part from a deliberately naive baseline (flat pattern
  weights, no regime filter). Ref 15's weak start beat a heavily
  community-optimized incumbent; more importantly here, a run seeded only from
  the champion inherits whatever D-1 baked into it, and running both gives a
  free check on whether the champion's structure is actually load-bearing.
- **Ties are not rankings.** When several elites fall inside each other's score
  spread (P2), do not collapse to an argmax — keep them all and let P4's bandit
  resolve the ordering with out-of-sample capital, which is the only
  discriminator that isn't already exhausted. Ref 15 saw two independent runs
  land on different, equally competitive solutions on a *low*-variance task.
- **Cost governance.** Reuse `research/cost_pilot.py`. Hard per-run $ cap, same
  pattern as `llm_thesis.max_usd_per_scan: 2.5`.

*Acceptance:* an end-to-end run produces ≥20 archive cells filled; ≥1 elite
clears the P1 gate on sealed holdout; the run's total $ and trial count are
recorded in the ledger; the naive-seeded and champion-seeded lineages are both
represented, and the report states whether they converged.

### P4 — Bandit allocation over live paper (2 sessions)

`research/bandit/` — Thompson sampling over archive elites, each arm getting a
paper sleeve. Posterior updates on realized excess R. Arms that decay get
starved automatically; this is the loop's answer to non-stationarity, and it's
the `AEL` result (fast bandit + slow reflection beat every fancier variant they
ablated).

Guardrails: max concurrent arms, per-arm capital cap, auto-retire on drawdown
breach, champion always retains a fixed control sleeve as the benchmark.

*Acceptance:* 4+ weeks of paper allocation with per-arm attribution vs the
champion control sleeve.

### P5 — Meta-loop (2 sessions)

Only after P4 has produced a real track record:

- Evolve the **proposer prompt** on outcome (`AlgoEvolve`'s outer loop) — the
  proposer's fitness is "expected gate-passing yield per dollar", not per-variant score.
- Distill accepted/rejected outcomes into a **reusable lesson store** the
  proposer reads (`EvoQuant`'s knowledge distillation; Rainier's
  `paper/reflection.py` is the seed of this).
- Widen auto-apply: graduate specific `(kind, severity)` tuples from
  recommend-only to auto-apply — but only for actions whose gate verdict is
  reproducible and whose rollback is a one-line `champion.yaml` revert.

---

## 5. Guardrails (non-negotiable)

| Risk | Control |
|---|---|
| Loop overfits itself | Trial ledger + DSR + PBO + sealed holdout + null test (P1). No promotion without all four. |
| Loop converges to one fragile peak | MAP-Elites diversity; worst-regime reward term; champion control sleeve always live. |
| LLM hallucinates edits | LLM never picks numbers or sees raw prices; typed mutation space; `ResearchThesis` validators (already built); every variant must compile + pass a smoke backtest before scoring. |
| Runaway spend | Per-run and per-day $ caps via `cost_pilot`; kill switch mirrors `max_usd_per_scan`. |
| Silent capital damage | Paper-only through P4; per-arm drawdown auto-retire; weekly Discord digest of every auto-applied change. |
| Unauditable drift | Every promotion writes `champion.yaml` version + parent + gate verdict into `history/`; one-line revert. |
| Reward hacking | Rewards pre-registered before search; changing the reward set is an L3 (human-gated) event, logged. |
| **Incumbent assumed clean** | The champion is re-scored through the gate as ledger entry zero (P1.6) and audited for the same hacks candidates are screened for. A baseline is not trustworthy because it is incumbent. |
| **Point scores mistaken for edge** | Evaluator returns `(score, spread, n_paths)`; candidates whose spread swamps their edge are rejected as *unresolved*, and ties are kept rather than ranked (P2, P3). |
| **Hand-screening doesn't scale** | Ref 15 admits missing hacks in results outside their expertise. Rainier's answer is a narrow typed proposal surface (P3 stage 1) plus automated leak checks, not reviewer diligence. |

---

## 6. What I recommend *against*

- **Don't widen auto-apply before P1.** Today's five executors are safe mostly
  because a human reads every insight. Autonomy before the gate is the failure mode.
- **Don't tune on the missed-winner sweep.** `DESIGN-qu100-llm-feedback-loop.md`
  D8 already rules it coverage-diagnostic-only (trailing point-to-point return,
  no entry timing, survivorship-biased). A recursive optimizer will happily
  optimize it if given the chance. Keep it out of every reward.
- **Don't evolve free-form strategy code first.** Stage the proposal surface.
- **Don't re-tune the 12 pattern weights on the current corpus.** Per D-1 that
  corpus scores beta. Re-tuning on it now bakes the artifact in.
- **Don't benchmark candidates against the champion's published numbers.**
  Re-score it first (P1.6). Comparing a gate-scored candidate to a
  pre-gate baseline measures the harness change, not the candidate.

---

## 7. Open questions for you

1. **Benchmark for excess return** — SPY only, or SPY + sector ETF residual?
   (Sector residual is more honest for a money-flow screener whose picks cluster
   by sector; it's ~half a session more work.)
2. **Sealed-holdout length** — 6 months costs you evaluation data now but is the
   only thing that will tell you in a year whether any of this worked. My
   default: 6 months, sealed until a promotion decision.
3. **Where does recursion get to touch capital?** My default: paper-only through
   P4, with a champion control sleeve, and a separate explicit decision before
   any live sizing.
4. **Is the futures/pin-bar side in scope?** The harness is generic, but P0's
   ruler fix is QU100-specific. Cheapest path is QU100 first, port after P2.

---

## 8. References

Ordered by the phase they inform. The anti-overfitting group is load-bearing and
peer-reviewed; the evolutionary group is recent, mostly self-reported in-sample,
and should be read as *architecture* references, not as evidence of edge.

### Anti-overfitting — read before P1 (the gate)

| # | Work | Link | Why it matters here |
|---|---|---|---|
| 1 | Bailey, Borwein, López de Prado & Zhu — *The Probability of Backtest Overfitting* (J. Computational Finance, 2016) | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253 | CSCV = the PBO estimator in the P1 gate. Shows plain hold-out is unreliable for backtests, which is why P1 needs more than a sealed window. |
| 2 | Bailey & López de Prado — *The Deflated Sharpe Ratio* (JPM 40(5), 2014) | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 | Deflates Sharpe by trial count + non-normality. Directly motivates the `research_trial` ledger: DSR is uncomputable without an honest trial count. |
| 3 | Bailey, Borwein, López de Prado & Zhu — *Pseudo-Mathematics and Financial Charlatanism* (Notices of the AMS 61(5), 2014) | https://www.ams.org/notices/201405/rnoti-p458.pdf | Minimum backtest length vs. number of trials. Short; the fastest way to see why `--sweep`/`--variations` with an unrecorded trial count is unsound. |
| 4 | López de Prado — *Advances in Financial Machine Learning*, ch. 7 | (book) | Purged K-fold + embargo. Applies directly to the 5/10/20d overlapping forward-return labels in the audit corpus. |

### Evolutionary / LLM search — read before P3–P5

| # | Work | Link | Why it matters here |
|---|---|---|---|
| 5 | LATSS — `vincent212/llm-assisted-trading-strategy-search` | https://github.com/vincent212/trading_strategy_evolution_agent | Closest running system to the P3 design: LLM mutates *structure only* and never picks a number; SciPy fits params; fitness is active return vs buy-and-hold with a shift-the-signal null test and sealed holdout years. Read the code, not just the README. |
| 6 | *QuantEvolve* (arXiv 2510.18569) | https://arxiv.org/abs/2510.18569 | Quality-diversity feature map over strategy type / risk / turnover — the source of the P3 MAP-Elites behavior descriptors. |
| 7 | *MadEvolve* (arXiv 2605.23007) | https://arxiv.org/abs/2605.23007 | AlphaEvolve-style evolution of a full trading system; §7 explicitly estimates p-hacking probability and compares against plain Claude Code as a search baseline. |
| 8 | *AEL* (arXiv 2604.21725) | https://arxiv.org/abs/2604.21725 | Thompson bandit (fast loop) + LLM reflection (slow loop) — the two-clock split behind P4/P5. Includes a "less is more" ablation. |
| 9 | *EvoQuant* (arXiv 2607.12455) | https://arxiv.org/abs/2607.12455 | Verifier-guided improvement of an *already-deployed* strategy plus experience distillation — the closest framing to Rainier's actual situation (a live champion, not a blank sheet). |
| 10 | *AlgoEvolve* (arXiv 2606.26173) | https://arxiv.org/abs/2606.26173 | Meta-evolution of the proposer prompt; the P5 reference. |
| 11 | *QuantaAlpha* (arXiv 2602.07085) | https://arxiv.org/abs/2602.07085 | Trajectory-level evolution with lineage/auditability and anti-crowding — informs the archive's lineage and diversity requirements. |

### Open-endedness / quality-diversity — the source of the P3 archive

None of these are finance papers. They are the upstream lineage that the
evolutionary trading work above borrows from, and they are where the "keep an
archive of interestingly different solutions rather than one champion" decision
in P3 actually comes from. Read 12 if you want to understand *why* P3 is an
archive and not a leaderboard.

| # | Work | Link | Why it matters here |
|---|---|---|---|
| 12 | Mouret & Clune (2015) — *Illuminating Search Spaces by Mapping Elites* | https://arxiv.org/abs/1504.04909 | The original MAP-Elites paper. The behavior-descriptor grid in P3 is this algorithm; §"illumination" is the argument for preferring a map over a maximum. |
| 13 | Zhang, Hu, Lu, Lange & Clune (2025) — *Darwin Gödel Machine* | https://arxiv.org/abs/2505.22954 | Self-modifying agent that keeps an archive and **empirically validates every self-modification against a benchmark**, with an explicit objective-hacking analysis. The closest existing precedent for the P1-gate-plus-P3-archive shape. |
| 14 | Clune (2019) — *AI-GAs: AI-Generating Algorithms* | https://arxiv.org/abs/1905.10985 | The position paper behind the whole "learn the improvement process, don't hand-design it" framing that P5 gestures at. |
| 15 | Recursive — *First Steps Toward Automated AI Research* (2026-06-11) | https://www.recursive.com/articles/first-steps-toward-automated-ai-research · https://github.com/recursive-org/first-steps-toward-automated-ai-research | Industrial automated research loop built on these principles. Notable for this plan: it validates results "for reward hacks and variance before treating improved performance as real progress", and it picked benchmarks with **tight feedback loops, clear metrics, low variance, and hardenable evaluators** — the precise properties financial backtests lack, which is the argument for P0/P1 in one sentence. |
