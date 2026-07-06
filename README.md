# Are Ethereum Whales Smart Money?

An empirical study of whether large on-chain Ethereum transactions (>$1M) predict short-term ETH price movements, and whether market sentiment moderates this relationship.

## Key Findings

### Whales have a small but real informational edge

Whale exchange withdrawals (buying signals) during bearish sentiment correctly predict price direction **65% of the time** at the 24-hour horizon. However, 60.5% of this is attributable to mean-reversion in bearish regimes — the whale-specific edge is approximately **+4.5%** above the base rate.

### The edge peaks at 12–24 hours, then decays

| Horizon | Whale hit rate | Base rate | Whale edge |
|---------|---------------|-----------|------------|
| 1h      | 52.1%         | 52.0%     | +0.1%      |
| 6h      | 58.5%         | 56.6%     | +1.9%      |
| **12h** | **64.6%**     | **58.8%** | **+5.8%**  |
| **24h** | **65.0%**     | **60.5%** | **+4.5%**  |
| 36h     | 62.6%         | 59.4%     | +3.2%      |
| 48h     | 58.1%         | 57.4%     | +0.8%      |
| 72h     | 61.6%         | 60.2%     | +1.4%      |
| 96h     | 60.9%         | 60.5%     | +0.4%      |
| 120h    | 60.0%         | 62.3%     | -2.3%      |

Whatever informational advantage whales have is fully incorporated into prices within 24 hours — consistent with high information efficiency in cryptocurrency markets at longer horizons.

### Sentiment regime is the key moderator

Whale actions alone do not predict price direction (overall hit rate ~49–51%). But whale actions **conditioned on sentiment** produce significant results:

**Whale withdrawals (buy signals):**
| Condition | 24h hit rate | p-value | Verdict |
|-----------|-------------|---------|---------|
| Negative funding rate | 65.0% | < 0.001 | Smart money |
| Fear (FnG ≤ 45) | 59.7% | < 0.001 | Smart money |
| Greed (FnG > 55) | 51.6% | < 0.001 | Slightly smart |
| Extreme greed (FnG > 75) | 45.5% | < 0.001 | Wrong |

Whales who withdraw during fear are contrarian smart money. Whales who withdraw during extreme greed are wrong — buying into euphoria.

### Average returns per signal

Whale withdrawals during negative funding rate:

| Horizon | Mean return | Median return | After 0.1% fees |
|---------|-----------|---------------|-----------------|
| 1h      | +0.08%    | +0.03%        | -0.02% (unprofitable) |
| 6h      | +0.48%    | +0.27%        | +0.38% |
| 24h     | +1.31%    | +0.84%        | +1.21% |

### ML price prediction gives modest results

A Random Forest model using all 22 features (whale, sentiment, price momentum) achieves 53.3% accuracy at the 6-hour horizon vs a 50.5% baseline (+2.9% edge). Feature importance shows price momentum and market sentiment dominate; whale features rank low — confirming that whale signal is directional and conditional, not a standalone predictor.

## Research Question

Do large on-chain Ethereum transactions (>$1M) systematically precede ETH price movements, and does market sentiment moderate this relationship?

**Answer:** Yes, but conditionally. Whale transactions alone are not predictive. However, whale withdrawals during bearish sentiment (negative funding rate or low Fear & Greed) correctly predict 24-hour price direction 65% of the time, with a 4.5% edge above the base rate. This informational advantage decays within 24 hours.

## Methodology

### Phase 1 — Whale Data Pipeline
- 392,517 large ETH transactions (Jan 2023 – Jun 2025) extracted via Dune Analytics
- 52,768 wallet addresses labelled using Etherscan labels and two open-source label datasets ([brianleect/etherscan-labels](https://github.com/brianleect/etherscan-labels), [dawsbot/eth-labels](https://github.com/dawsbot/eth-labels))
- Label coverage: 67.6% of transactions have at least one identified address
- MEV bot candidates flagged (not removed) for sensitivity analysis

### Phase 2 — Transaction Classification
- Rule-based labelling for known wallets: exchange deposit, exchange withdrawal, DeFi interaction, wallet-to-wallet
- Random Forest classifier trained on labelled transactions to predict categories for the remaining 32% unknown-to-unknown transactions
- Classifier accuracy on time-based hold-out: 71%

### Phase 3 — Sentiment Pipeline
- **News sentiment:** 5,906 Bitcoin news articles (Kaggle dataset) scored with VADER and RoBERTa, aggregated into 3,474 hourly bins
- **Fear & Greed Index:** Daily composite score (0–100) from alternative.me — captures volatility, volume, social media, BTC dominance
- **Binance funding rate:** 8-hourly real-money sentiment — positive = longs pay shorts (bullish), negative = shorts pay longs (bearish)

### Phase 4 — Event Study (Core Analysis)
- For each whale transaction, compute forward ETH returns at 1h, 6h, 12h, 24h, 36h, 48h, 72h
- Measure **hit rate**: did the whale's action correctly predict price direction?
  - Exchange deposit (sell signal): hit = price dropped
  - Exchange withdrawal (buy signal): hit = price rose
- Condition on sentiment regime (Fear & Greed, funding rate)
- Compare to base rate to isolate the whale-specific edge
- Statistical significance via binomial test

### Walk-Forward Validation (ML Model)
- Train on months 1–N, predict month N+1, slide forward — no look-ahead bias
- Logistic Regression and Random Forest across 24 walk-forward folds
- StandardScaler fit on training data only

## Data Sources

| Source | Data | Records | Cost |
|--------|------|---------|------|
| Dune Analytics | Whale transactions (>$1M) | 392,517 | Free tier |
| Binance API | Hourly ETH prices | 21,888 | Free |
| Binance API | ETH funding rates (8-hourly) | 2,736 | Free |
| alternative.me | Fear & Greed Index (daily) | 3,071 | Free |
| Kaggle | Bitcoin news headlines | 5,906 (overlap window) | Free |
| Etherscan labels (GitHub) | Wallet address labels | 52,768 | Free |

## Limitations

1. **Survivorship bias in wallet labels.** We only label wallets that are publicly known. Some "unknown" wallets may be exchange or institutional wallets we cannot identify, which could dilute or distort the signal.

2. **Whale-specific edge is modest.** The +4.5% edge at 24h is statistically significant but economically marginal. After transaction costs, slippage, and execution latency, live trading profitability is uncertain.

3. **Negative funding regime is rare.** Only ~9% of hours in our dataset have negative funding. A strategy relying on this condition would be idle most of the time.

4. **Backtested, not live-tested.** Historical results do not guarantee future performance. If market participants begin following whale signals, the edge would be arbitraged away.

5. **On-chain latency.** Whale transactions are visible on-chain after confirmation (~12 seconds), but monitoring, processing, and executing a trade in response takes additional time. Some of the edge may be consumed by this latency.

6. **News sentiment is sparse.** Average 1.7 articles per hour — many hours have no coverage. Market-derived sentiment (Fear & Greed, funding rate) proved far more useful than news headlines.

7. **ETH trended upward over the sample period (Jan 2023 – Jun 2025).** All raw forward returns are positively biased. We address this by comparing to base rates, but the results may not generalise to prolonged bear markets.

8. **No causal claim.** A $1M transaction does not move ETH's price. We test whether whales predict direction, not whether they cause it.

## Repository Structure

```
whale_signals/
├── data/
│   ├── raw/                  # Raw data (gitignored)
│   ├── processed/            # Cleaned datasets (gitignored)
│   └── reference/            # Curated wallet labels (version-controlled)
├── src/
│   ├── data/                 # Dune client, price fetcher, sentiment fetcher
│   ├── features/             # Feature engineering, Phase 4 feature matrix
│   ├── models/               # Transaction classifier, price predictor
│   ├── sentiment/            # VADER scorer, hourly aggregator
│   └── analysis/             # Event study (hit rates, conditioned analysis)
├── scripts/                  # Pipeline runners
├── tests/                    # Unit tests
├── docs/                     # Design notes, project arc
└── results/                  # Charts and model artefacts
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run sentiment pipeline
python scripts/run_sentiment_pipeline.py

# 3. Build Phase 4 feature matrix
python scripts/run_phase4_features.py

# 4. Run ML model (walk-forward validation)
python scripts/run_phase4_model.py

# 5. Run event study (core analysis)
python scripts/run_event_study.py
```

Note: Raw whale data requires Dune Analytics API access. Pre-processed data is not included in the repository due to file size.
