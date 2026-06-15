# Full Project Arc

## Research Question

Do large on-chain Ethereum transactions (>$1M) predict short-term ETH price
movements, and does social media sentiment moderate this relationship?

## Phase Overview

Each phase feeds into the next. Understanding the full arc is essential before
touching any single phase.

### Phase 1 -- Whale Data Pipeline (COMPLETE)

Extract large ETH transactions from the blockchain via Dune Analytics. Enrich
with wallet labels (exchange, DeFi, unknown). Flag MEV bots.

- Output: `data/processed/whale_txs.csv` with labelled, cleaned transactions.
- Key columns: `timestamp_utc`, `from_address`, `to_address`, `from_category`,
  `to_category`, `eth_value`, `usd_value`, `gas_price_gwei`, `gas_used`,
  `is_contract_call`, `is_mev_candidate`, `mev_flag_reason`

Key decisions documented in `docs/design_notes.md`:
- Dune Analytics over Etherscan (chain-wide SQL vs per-address API)
- WETH as ETH price proxy (no native ETH in Dune's prices.usd table)
- Internal transactions out of scope (ethereum.traces not queried)
- MEV candidates flagged, not deleted (for Phase 4 sensitivity analysis)

### Phase 2 -- Transaction Classification (COMPLETE)

The raw labels (exchange_deposit, exchange_withdrawal, defi_interaction,
wallet_to_wallet) are derived mechanically from address labels. The classifier
learns to generalise this to unlabelled (unknown->unknown) transactions using
features like gas price, transaction size, sender history, time of day.

- Output: each transaction gets a predicted category + probability score.
- Key files: `src/features/feature_engineer.py`,
  `src/models/transaction_classifier.py`

Results on real data (292,445 transactions, 2023-01-01 to 2024-12-31):
- 76% of transactions are unknown->unknown (classifier is load-bearing).
- Label coverage: 11.3% from-address labelled, 14.0% to-address labelled.
- Category distribution: 222,229 wallet_to_wallet, 37,895 exchange_deposit,
  29,336 exchange_withdrawal, 2,985 defi_interaction.
- Classifier accuracy on time-based hold-out: 71%.
  DeFi: perfect. Deposits: 85% precision / 54% recall.
  Withdrawals: 63% precision / 89% recall.
- Top features: sender_prior_tx_count (29%), log_gas_used (28%),
  is_contract_call (19%).
- Look-ahead safe: sender history uses cumcount (only prior rows).

### Phase 3 -- Sentiment Pipeline (COMPLETE)

Bitcoin news headlines (Kaggle dataset, 11,295 articles Oct 2021 - Sep 2024)
scored with VADER sentiment. Filtered to whale data overlap window (Jan 2023 -
Sep 2024, 5,906 articles), aggregated into 3,474 hourly bins.

Data source: Kaggle "Sentiment Analysis of Bitcoin News (2021-2024)" dataset.
Chosen over GDELT (rate-limited, IP-banned), Reddit (API paywalled),
CryptoPanic (API paywalled), CoinDesk (API retired).

Two sentiment signals available:
- VADER compound score (lexicon-based, transparent, reproducible)
- RoBERTa score (pre-computed in dataset, DistilRoBERTa fine-tuned on
  financial news)
- Correlation between the two: 0.313 (weak agreement -- complementary signals)

- Output: `data/processed/hourly_sentiment.csv` (hourly aggregated sentiment),
  `data/processed/bitcoin_news_scored.csv` (per-article scores).
- Key files: `src/sentiment/vader_scorer.py`, `src/sentiment/aggregator.py`,
  `scripts/run_sentiment_pipeline.py`
- Caveat: 1.7 articles/hour average is sparse. Phase 4 must forward-fill
  sentiment for hours with no articles (with decay).
- Why this matters: whale movements in isolation are noisy. If a whale deposits
  to an exchange during strongly negative sentiment, the selling pressure signal
  is stronger than the same transaction during positive sentiment.

### Phase 4 -- Price Impact Prediction (THE PAYOFF)

All prior phases combine. Features fed into the final model:
- Whale transaction category (from Phase 2 classifier)
- Transaction size and gas (from Phase 1)
- Hourly sentiment score (from Phase 3)
- Recent price features (rolling returns, volatility)

Target variable: ETH price direction at t+24h (binary: up or down).
Walk-forward validation: model trained only on data available before each
prediction. No look-ahead.

- Output: directional accuracy, edge over random baseline, P&L simulation.

### Phase 5 -- Evaluation and Write-up

Honest reporting of findings. Where does the signal exist? Which transaction
categories drive it? Does sentiment improve or not? What are the limitations?

- Output: `results/charts/`, `docs/findings.md`, final README.

## Where ML Appears

| Phase   | ML Component              | Type                                          |
|---------|---------------------------|-----------------------------------------------|
| Phase 2 | Transaction classifier    | Supervised classification (Random Forest)     |
| Phase 3 | Sentiment scoring         | Pre-trained NLP (VADER, optional FinBERT)     |
| Phase 4 | Price impact predictor    | Supervised classification (XGBoost/LogReg)    |

Phase 2 ML is in service of Phase 4. The classifier is not the end goal -- it
is a feature engineering step that makes Phase 4 possible by extending label
coverage to unknown->unknown transactions.

## Session Handoff Notes

Update this section at the end of each working session.

**Last session: 2026-06-15**
- Phase 3 complete (105 tests passing, 10 skipped).
- Sentiment data: Kaggle Bitcoin news dataset, 5,906 articles in overlap window.
- Two sentiment scores: VADER (ours) and RoBERTa (pre-computed). Correlation 0.313.
- Hourly sentiment: 3,474 bins saved to data/processed/hourly_sentiment.csv.
- Dune API credits exhausted. DO NOT run dune_client.py or whale_fetcher.py.
- Next step: Phase 4 -- price impact prediction. Need ETH price data (hourly),
  merge whale transactions + sentiment + price, build predictive model.
