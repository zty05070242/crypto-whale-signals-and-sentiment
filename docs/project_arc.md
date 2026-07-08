# Full Project Arc

## Research Question

Do large on-chain Ethereum transactions (>$1M) predict short-term ETH price
movements, and does market sentiment moderate this relationship?

## Phase Overview

Each phase feeds into the next. Understanding the full arc is essential before
touching any single phase.

### Phase 1 -- Whale Data Pipeline (COMPLETE)

Extract large ETH transactions from the blockchain via Dune Analytics. Enrich
with wallet labels (exchange, DeFi, unknown). Flag MEV bots.

- Output: `data/processed/whale_txs.csv` with labelled, cleaned transactions.
- 646,442 transactions, Jan 2023 -- Jul 2026.
- Key columns: `timestamp_utc`, `from_address`, `to_address`, `from_category`,
  `to_category`, `eth_value`, `usd_value`, `gas_price_gwei`, `gas_used`,
  `is_contract_call`, `is_mev_candidate`, `mev_flag_reason`

Key decisions documented in `docs/design_notes.md`:
- Dune Analytics over Etherscan (chain-wide SQL vs per-address API)
- WETH as ETH price proxy (no native ETH in Dune's prices.usd table)
- Internal transactions out of scope (ethereum.traces not queried)
- MEV candidates flagged, not deleted (for sensitivity analysis)

Wallet labels expanded from 30 to 52,768 addresses using two open-source
datasets (brianleect/etherscan-labels, dawsbot/eth-labels). Label coverage:
62.8% of transactions having at least one identified address.

### Phase 2 -- Transaction Classification (COMPLETE)

Rule-based labels (exchange_deposit, exchange_withdrawal, defi_interaction,
wallet_to_wallet) derived from wallet labels. Random Forest classifier
trained on labelled transactions to predict categories for remaining
unknown-to-unknown transactions.

- Output: each transaction gets a predicted category + probability score.
- Key files: `src/features/feature_engineer.py`,
  `src/models/transaction_classifier.py`

Category distribution (full dataset):
- wallet_to_wallet: 321,257 (49.7%)
- exchange_deposit: 181,105 (28.0%)
- exchange_withdrawal: 124,772 (19.3%)
- defi_interaction: 19,308 (3.0%)

Classifier accuracy on time-based hold-out: 71%.

### Phase 3 -- Sentiment Pipeline (COMPLETE)

Three sentiment sources:

1. **News sentiment:** Kaggle Bitcoin news dataset (5,906 articles in overlap
   window), scored with VADER and RoBERTa. Proved to be a weak signal.

2. **Fear & Greed Index:** Daily composite score 0-100 from alternative.me.
   3,075 records.

3. **Binance funding rate:** 8-hourly, positive = longs pay shorts (bullish),
   negative = shorts pay longs (bearish). 3,851 records.

Market-derived sentiment (FnG + funding rate) proved far more useful than
news headlines for conditioning whale signals.

### Phase 4 -- Event Study: Are Whales Smart Money? (COMPLETE)

The core analysis. Walk-forward event study with threshold sensitivity.

**Methodology:**
- For each whale transaction, compute forward ETH return at 1h, 6h, 24h.
- Measure hit rate and compare to base rate under same market conditions.
- Walk-forward by year: 2023, 2024, 2025, 2026 analysed independently.
- Threshold sensitivity: $1M, $2M, $5M, $10M minimum transaction size.
- Condition on sentiment regime (Fear & Greed, funding rate).

**Key findings:**

1. **Deposit edge is persistent and growing.** Unconditional deposit hit rate
   at 24h: +3.9% edge in 2026 (out-of-sample), holding consistently at every
   threshold from $1M+ to $10M+. A narrower claim -- $10M+ deposits during
   extreme greed hitting 78.3% (+12.5% edge) -- was later found not to
   survive scrutiny for overlapping observations (see the Priority 1
   resolution notes further down this file). Do not cite the 78.3% figure as
   a finding; it is kept in the README only as a worked example of the
   overlap problem.

2. **Withdrawal edge has decayed. DeFi maturation is the evidenced explanation.**
   Was +4.7% to +10.1% in 2023-2024, collapsed to zero in 2025-2026 at all
   thresholds -- including $10M+, which rules out "just small-actor dilution"
   and points to withdrawals no longer meaning directional buying (staking,
   LP, L2 bridging instead). This is backed directly by our own threshold data.

3. **Alpha decay is asymmetric, and dilution should have weakened the deposit
   signal but didn't.** As ETH's price rose, a fixed $1M threshold captured
   progressively less committed actors (~833 ETH in 2023 vs ~250 ETH in 2026),
   yet the deposit edge grew despite this headwind -- an argument against the
   edge being a simple composition effect. Why the deposit signal specifically
   wasn't arbitraged away is speculative and unevidenced: one untested
   hypothesis is that whale-watching tools broadcast bullish activity more than
   bearish activity. A brief search found only an indirect, tangential data
   point (Whale Alert tweets about USDT minting amplify BTC price reactions),
   not a direct test of buy/sell broadcast asymmetry. Treat this as plausible,
   not proven -- unlike finding #2, it is not backed by our own data.

4. **Unconditional whales are not smart money.** Full-dataset hit rates near 50%.
   The edge is conditional on sentiment regime and transaction size.

5. **The edge is real but comes with real pain: maximum adverse excursion.**
   For deposit signals that eventually paid off, the average drawdown before
   they did grows sharply with horizon: 2.7% at 1 week to 20.1% at 6 months
   (unconditional), with the worst 10% of "correct" 6-month trades seeing a
   54.4% adverse move first. Confirms Limitation #6 (now solved, partially --
   MAE is measured, but no stop-loss RULE is simulated).

**Secondary finding (ML model):**
Walk-forward Random Forest with 22 features achieves 53.3% accuracy at 6h.
Feature importance shows price momentum dominates; whale features rank low.

- Key files: `src/analysis/event_study.py`, `scripts/run_event_study.py`,
  `scripts/run_walk_forward.py`, `scripts/run_threshold_sensitivity.py`,
  `scripts/run_drawdown_analysis.py`
- ML files: `src/features/phase4_features.py`, `src/models/price_predictor.py`

### Phase 5 -- Dashboard, Deployment, and Write-up (COMPLETE)

Interactive Streamlit dashboard, deployed publicly, and research-format README.

**Live dashboard:** https://crypto-whale-signals-and-sentiment-lkhygb3594bbrogn23qbps.streamlit.app/

**Dashboard architecture:**
The dashboard does not read the raw 646k-row dataset at runtime. Instead
`scripts/build_dashboard_data.py` runs once locally, crunching the full
dataset into `app/dashboard_data.json` (~480 KB): every hit rate, edge, and
histogram the dashboard renders, pre-computed and rounded. `app/dashboard.py`
imports only `streamlit` and `plotly` and reads this JSON, so it renders
instantly and needs no pandas/numpy/web3 at runtime. This also solves
deployment: the raw CSV is 187 MB and gitignored (too large and not meant for
git), but the 480 KB JSON commits cleanly and is all Streamlit Community Cloud
needs. Re-run the build script whenever the underlying data changes, then
commit and push the regenerated JSON.

**Dashboard content (7 sections):**
1. Deposit edge across horizons (1h to 6m) -- shows the edge compounding over time.
2. Yearly stability / alpha decay (deposits vs withdrawals), horizon-selectable.
3. Threshold sensitivity ($1M to $10M+), horizon-selectable.
4. Sentiment-conditioned hit rates across 7 regimes (fixed at 24h).
5. Return distribution of deposits, with a regime dropdown (fixed at 24h) --
   shows the signal is conditional: unconditional deposits sit near a coin
   flip (50.5%); the bearish skew only appears in specific regimes.
6. Deposit vs withdrawal edge asymmetry by year, horizon-selectable.
7. Limitations (backtested-not-live, thin edge at short horizons, overlapping
   long-horizon windows, no stop-loss modelling, threshold dilution, dead
   withdrawal signal).

Horizon selection was deliberately scoped to sections 2/3/6 and not 4/5:
those two already carry a 7-regime chart and a regime dropdown, and stacking a
horizon dimension on top would push some combinations into small-n territory,
which is exactly the "overlapping windows overstate significance" risk in the
limitations section.

Visual theme: dark "quant terminal" styling (mono fonts, muted TradingView-style
green/red, bordered metric cards) via `.streamlit/config.toml` and inline CSS --
chosen deliberately over a neon "degen" look, to stay consistent with the
project's honest-reporting framing while still reading as crypto-native product.

**Deployment:** pushed to GitHub, deployed on Streamlit Community Cloud
(`app/requirements.txt` holds the minimal streamlit+plotly dependency set for
the cloud build). Repo was renamed on GitHub from `whale_signals` to
`crypto-whale-signals-and-sentiment`; the local remote still points at the old
name but GitHub's redirect keeps `git push` working.

- Dashboard: `app/dashboard.py`, `app/dashboard_data.json`, `app/requirements.txt`
- Build script: `scripts/build_dashboard_data.py`
- Theme: `.streamlit/config.toml`
- README with full results, methodology, and limitations.
- Walk-forward results: `results/walk_forward_results.csv`
- Threshold sensitivity: `results/threshold_sensitivity.csv`

## Where ML Appears

| Phase   | ML Component              | Type                                          |
|---------|---------------------------|-----------------------------------------------|
| Phase 2 | Transaction classifier    | Supervised classification (Random Forest)     |
| Phase 3 | Sentiment scoring         | Pre-trained NLP (VADER)                       |
| Phase 4 | Price impact predictor    | Supervised classification (RF + LogReg)       |
| Phase 4 | Event study               | Statistical testing (binomial test)           |

## Session Handoff Notes

Update this section at the end of each working session.

**Last session: 2026-07-07**
- Data expanded to 646,442 transactions (Jan 2023 -- Jul 2026) via second
  Dune account. Prices, funding rates, FnG all updated through Jul 2026.
- Walk-forward event study by year completed. Key discovery: withdrawal edge
  decayed from 2023 to 2026, deposit edge grew. Alpha decay is asymmetric.
- Threshold sensitivity analysis: deposit edge scales with transaction size.
  $10M+ deposits during extreme greed = 78.3% hit rate (+12.5% edge).
- DeFi dilution hypothesis: withdrawal signal broke down because DeFi maturation
  changed what withdrawals mean (non-directional: staking, LP, L2 bridging).
- README rewritten with walk-forward results and threshold sensitivity.
- Scripts: run_walk_forward.py, run_threshold_sensitivity.py added.
- event_study.py vectorised (was timing out on 646k rows with .apply()).
- Dashboard fully rewritten: dark quant-terminal theme, deposit-centric
  (not withdrawal-centric), 7 sections covering horizons/years/thresholds/
  sentiment/distribution/asymmetry/limitations. Forward returns vectorised.
- Deployment architecture built: scripts/build_dashboard_data.py pre-computes
  every dashboard number into app/dashboard_data.json (~480 KB), so the app
  can deploy to Streamlit Community Cloud without the 187 MB gitignored raw
  dataset. app/dashboard.py rewritten to read only this JSON (streamlit +
  plotly only, no pandas at runtime).
- Deployed to Streamlit Community Cloud:
  https://crypto-whale-signals-and-sentiment-lkhygb3594bbrogn23qbps.streamlit.app/
- Fixed a stale-@st.cache_data bug that crashed the live app after a redeploy
  (KeyError: 'dist_conditions'): load_data() now keys its cache on the data
  file's mtime so a new JSON always busts the old cached dict.
- Added a return-distribution regime selector (section 5) and a horizon
  selector for sections 2/3/6, so yearly stability, threshold sensitivity, and
  the asymmetry chart can all be viewed at 1h through 6m, not just 24h.
- Repo renamed on GitHub: whale_signals -> crypto-whale-signals-and-sentiment.
- Fred requested a 4-part rigour pass on the write-up. Tackled 3 of 4 this
  session (order: drawdown first, since its output fed the Discussion
  rewrite; then the two Discussion edits together):
  - **Drawdown / maximum adverse excursion (new script, done).**
    `scripts/run_drawdown_analysis.py` computes, for each long-horizon
    deposit signal, the worst price move against the position before the
    final outcome (a forward rolling max via the reverse-rolling-max pandas
    idiom, unit-tested before running on real data). Added README Section 8
    and softened Limitation #6 accordingly. Cross-checked against existing
    published numbers (extreme-greed 6-month blended return -23.1% here vs
    -22.6% already in the README) to confirm no bug before writing it up.
  - **Dilution-strengthens-the-case argument (done).** Moved out of
    Limitations into Discussion and the abstract: a fixed $1M threshold
    getting diluted over time should have weakened the deposit edge; instead
    it grew. Argues against the edge being a composition effect.
  - **Evidenced vs speculative reframing (done).** DeFi maturation (backed by
    our own threshold data: signal is dead at every size) is now explicitly
    the primary explanation for withdrawal decay. The whale-tool-broadcasting
    idea for why deposit survived is now explicitly labelled speculative and
    untested. Ran one cheap web search for supporting evidence; found only an
    indirect, tangential data point (Whale Alert/USDT-minting amplification
    study) which is cited but clearly hedged, not treated as confirmation.
  - **Deferred: effective sample size / overlapping-event correction
    (Priority 1, NOT yet implemented).** Investigated this first and found it
    is a bigger issue than initially framed: clustering whale events into
    non-overlapping calendar-period blocks (robustness-checked across
    multiple block-anchor offsets) shows even the UNCONDITIONAL 24h claim
    loses significance (raw 50.5% hit rate, p=1.1e-4, n=180,963 -> effective
    49.4%, p=0.70, n=1,283 independent days). The abstract's flagship 78.3%
    figure ($10M+ deposits, extreme greed, 2025 only, N=350) comes from just
    10 distinct calendar days across 5 episodes -- but is directionally
    robust under the correction (80.0% effective hit rate, broad-based across
    8 of 10 days, not one outlier), just underpowered (p=0.109). The honest
    dividing line is not horizon length (as originally proposed) but breadth
    of claim: broad/unconditional numbers do not survive scrutiny at any
    horizon; narrow/large-ticket/regime-specific numbers hold up directionally
    but cannot clear conventional significance with only a handful of
    independent market episodes in 3.5 years of data. Fred wants to decide
    the correction methodology (block-clustering, the simple method already
    computed, vs. Newey-West/HAC-corrected regression, the more standard but
    heavier academic tool) before this gets implemented and written up.
- **Priority 1: final resolution, after several false starts (read this one,
  not any earlier draft of this note -- the arc matters for next time).**

  The investigation went through three methods, in order of decreasing
  sophistication, each abandoned for the same underlying reason:

  1. `scripts/run_hac_regression.py` -- reframed the overlap problem as a
     regression (forward return ~ log deposit $ volume per hour, with
     Newey-West HAC standard errors, tested at 3 lag choices per cell to
     catch that the theoretically-motivated lag is itself unstable at long
     horizons). Confirmed the 78.3% flagship claim does not survive
     (coefficient ~0, wrong sign) and surfaced a new "3-month, extreme greed,
     deposit volume" finding that held up across every lag tested. **Deleted.**
     After walking through exactly what the script did, Fred concluded he did
     not understand it well enough to defend it in an interview and asked to
     ditch it. Right call, not argued out of.
  2. A block-partition check (grouping events into non-overlapping calendar
     periods so clustered same-day transactions count once, not many times)
     -- confirmed the same finding via a method Fred does understand and
     argued through himself ("is the 3-month horizon really only 1 episode",
     "if a whale wins, doesn't that count towards the hit rate" -- both good,
     correct challenges he raised unprompted). Found the 78.3% claim rests on
     just 10 genuinely independent calendar days (p=0.109, not significant).
  3. `scripts/run_daily_deposit_frequency.py` -- an independently-designed
     third method (busy vs quiet periods for deposit activity, block length
     matched to the horizon being tested). Two real bugs were caught and
     fixed via synthetic unit tests before trusting it on real data (a naive
     one-row-per-calendar-day design didn't account for busy days clustering
     together, which reintroduces overlap at long horizons; a pd.cut binning
     bug silently folded a dropped incomplete block's deposits into the prior
     block). The correctly-built version, tested at the only horizon with
     enough non-overlapping blocks to mean anything (1 month, ~42 blocks),
     came back a clean null (-6.5% edge, p=0.55).

  **Then Fred proposed a clean final scope, and this is what actually
  shipped:** ditch every statistical correction method from the write-up
  entirely (no effective-N language, no block-partition description, no
  regression) -- not because any of them were wrong, but because describing
  a correction technique in the README creates a defend-it-in-an-interview
  obligation that isn't worth taking on. Keep only the plain-English
  Limitation #1 ("long-horizon observations overlap... report the edge, not
  the p-value"), which was already accurate and needs no methodology behind
  it to defend. Fix the abstract/Original Contributions so the 1-month and
  6-month figures aren't stated with the same unqualified confidence as the
  24h figure, consistent with that limitation -- a wording change, not new
  analysis. The 78.3% Discussion passage was rewritten the same way: states
  plainly that the claim rests on a small, clustered slice of history, with
  no mention of calendar periods, effective N, or p-values.

  `scripts/run_daily_deposit_frequency.py` and `run_hac_regression.py` are
  both deleted; neither was ever committed, so there is no git history to
  clean up. `results/daily_deposit_frequency.csv` and
  `results/hac_regression_results.csv` were also removed (gitignored anyway).

  **One thing Fred asked for and was explicitly declined, worth recording
  precisely so a future session does not accidentally do it:** presenting
  the UN-spaced version of the busy-period test (which showed a ~20-26% edge,
  because it has the same overlap problem as the original 78.3% claim) as a
  legitimate finding, either outright or with the overlap caveat relegated to
  a separate Limitations bullet disconnected from the specific number. Both
  versions were refused for the same reason: the project already demonstrates,
  with its own data, that this exact overstatement mechanism is real and
  large (p=2x10^-27 naive vs p=0.11 corrected for the same claim) -- using an
  un-spaced number after establishing that would be knowingly misleading, not
  a scope or style choice. If a future session is asked to revisit this test,
  any result must either use properly-spaced (non-overlapping) blocks, or be
  presented with its overlap caveat immediately adjacent to the number, not
  filed separately.

  **Lesson for future sessions:** the right level of rigour for this project
  is "plain English, no specific correction technique asserted" -- not
  because more rigorous methods are wrong, but because Fred needs to be able
  to defend everything in the document himself, and a technique he can't
  explain is a liability regardless of whether it's statistically correct.
- Next steps: nothing outstanding on Priority 1 -- closed at the "honest
  plain-English disclosure" level, deliberately, after trying three more
  sophisticated alternatives first. Still open: whether section 4 (sentiment)
  or section 5 (distribution) on the dashboard ever warrant their own horizon
  control once more data reduces the small-n risk at long horizons.
