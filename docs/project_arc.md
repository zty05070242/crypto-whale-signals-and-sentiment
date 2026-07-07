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
   at 24h: +3.9% edge in 2026 (out-of-sample). During extreme greed, $10M+
   deposits hit 78.3% (+12.5% edge). Edge scales with transaction size.

2. **Withdrawal edge has decayed.** Was +4.7% to +10.1% in 2023-2024, collapsed
   to zero in 2025-2026 at all thresholds. DeFi maturation changed what
   withdrawals mean (staking, LP, L2 bridging vs directional buying).

3. **Alpha decay is asymmetric.** Buy signals decayed; sell signals strengthened.
   Consistent with confirmation bias: participants monitor bullish whale activity
   but ignore bearish signals, leaving the deposit edge un-arbitraged.

4. **Unconditional whales are not smart money.** Full-dataset hit rates near 50%.
   The edge is conditional on sentiment regime and transaction size.

**Secondary finding (ML model):**
Walk-forward Random Forest with 22 features achieves 53.3% accuracy at 6h.
Feature importance shows price momentum dominates; whale features rank low.

- Key files: `src/analysis/event_study.py`, `scripts/run_event_study.py`,
  `scripts/run_walk_forward.py`, `scripts/run_threshold_sensitivity.py`
- ML files: `src/features/phase4_features.py`, `src/models/price_predictor.py`

### Phase 5 -- Dashboard, Deployment, and Write-up (COMPLETE)

Interactive Streamlit dashboard, deployed publicly, and research-format README.

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
- Deployed to Streamlit Community Cloud (user-side deploy via
  share.streamlit.io; live URL not yet recorded in this repo -- add it here
  once confirmed).
- Fixed a stale-@st.cache_data bug that crashed the live app after a redeploy
  (KeyError: 'dist_conditions'): load_data() now keys its cache on the data
  file's mtime so a new JSON always busts the old cached dict.
- Added a return-distribution regime selector (section 5) and a horizon
  selector for sections 2/3/6, so yearly stability, threshold sensitivity, and
  the asymmetry chart can all be viewed at 1h through 6m, not just 24h.
- Repo renamed on GitHub: whale_signals -> crypto-whale-signals-and-sentiment.
- Next steps: record the live Streamlit Cloud URL in this file and the
  README once available; consider whether section 4 (sentiment) or section 5
  (distribution) ever warrant their own horizon control once more data
  reduces the small-n risk at long horizons.
