# Are Ethereum Whales Smart Money? An Event Study of On-Chain Signals and Sentiment

## Abstract

This study investigates whether large on-chain Ethereum transactions predict short-term and medium-term ETH price movements, and whether market sentiment moderates this relationship. Using an event study methodology on **646,442 whale transactions** over **3.5 years** (Jan 2023 - Jul 2026), we measure the directional hit rate of whale exchange deposits (sell signals) and withdrawals (buy signals) across nine time horizons (1 hour to 6 months), four transaction size thresholds ($1M to $10M+), seven sentiment regimes, and four independent yearly samples.

**Key findings:**

The clearest result: the deposit (sell) signal's edge grows monotonically from +1.3% at 24 hours to +12.4% at 6 months, and this pattern was set using 2023-2025 data and confirmed, unchanged, when tested against 2026 data collected afterward.

1. **Whale deposits (sell signals) show a persistent, growing edge that strengthens with horizon, and holds at every transaction size.** The unconditional deposit edge grows from +1.3% at 24h to +4.8% at 1 month to +12.4% at 6 months. This growth over time shows up consistently whether we look at $1M+ or $10M+ deposits alone: both move from roughly flat in 2023 to +2.9% to +3.9% by 2026. Whale sellers think in weeks and months, not hours. (A separate, narrower claim we initially reported, that $10M+ deposits during extreme greed showed a 78.3% hit rate, did not hold up under closer testing; see Discussion.)

2. **Whale withdrawals (buy signals) lost their edge entirely, and this holds at every transaction size too.** In 2023-2024, withdrawals during negative funding showed +4.7% to +10.1% edge; by 2025 that had faded close to zero at every threshold, and by 2026 it was negative at every threshold from $1M+ to $10M+, ruling out the simple explanation that small, less sophisticated actors were just diluting the signal. At longer horizons, withdrawals become actively wrong: -6.1% edge at 1 month, -12.9% at 6 months. We attribute this to DeFi maturation: withdrawals increasingly represent staking, liquidity provision, and L2 bridging rather than directional buying.

3. **Alpha decay is asymmetric, and a headwind that should have weakened the deposit signal did not.** As ETH's price rose, a fixed $1M threshold captured progressively smaller, less committed sellers (~833 ETH in 2023 vs ~250 ETH in 2026), dilution that should work against the deposit edge. Instead the edge grew, which argues against a simple composition-effect explanation. One explanation for the reason why the deposit signal avoided being arbitraged away, unlike withdrawals, is that whale-watching tools amplify bullish activity more than bearish.

5. **A second, independent method reaches a compatible conclusion.** A walk-forward validated Random Forest model (24 folds, no single-shot fitting) finds a modest but real edge over baseline (+2.0% to +2.9% at short horizons), and its feature importance ranks whale-specific features near the bottom, with price momentum and market sentiment dominating. A completely different technique, applied independently, agrees with the event study: whale activity is a real but conditional signal, not a standalone predictor.

6. **The edge is real, and it comes with a real cost.** For long-horizon deposit signals that eventually paid off, we measured the maximum adverse excursion: the worst unrealised loss before the signal worked. This grows sharply with horizon, from 2.7% at 1 week to 20.1% at 6 months on average, with the worst 10% of "correct" 6-month trades seeing a 54.4% adverse move first. The edge is not free money; collecting it means surviving drawdowns most traders cannot tolerate.

---

## Table of Contents

- [Background](#background)
- [Original Contributions](#original-contributions)
- [Data Sources](#data-sources)
- [Methodology](#methodology)
- [Results: Unconditional Hit Rates](#1-unconditional-hit-rates)
- [Results: Conditioned on Sentiment](#2-conditioned-on-sentiment-regime)
- [Results: Yearly Stability](#3-yearly-stability-analysis)
- [Results: Threshold Sensitivity](#4-threshold-sensitivity)
- [Results: Long Horizons](#5-long-horizon-analysis-1h-to-6-months)
- [Results: Deposits by Year at Long Horizons](#6-deposit-edge-by-year-at-long-horizons)
- [Results: ML Model](#7-ml-model-secondary-analysis)
- [Results: Drawdown During the Holding Period](#8-drawdown-during-the-holding-period)
- [Discussion](#discussion)
- [Limitations](#limitations)
- [Dashboard](#interactive-dashboard)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)

---

## Background

In traditional equity markets, institutional positioning is reported quarterly (13F filings) with significant lag. In contrast, public blockchains like Ethereum have every transaction visible in near-real-time. This creates a unique opportunity to study large-holder behaviours in real time.

Prior work on "whale watching" treated it as a price prediction problem: feed whale data into an ML model and attempt to forecast returns. This approach conflates multiple signals and rarely isolates the whale-specific contribution. This new approach now differs in three ways:

1. **Event study methodology**: we directly measure whether whale actions predict direction, rather than building a black-box predictor.
2. **Sentiment conditioning**: we test whether market regime (Fear & Greed Index, futures funding rate) moderates the whale signal.
3. **Yearly stability analysis.** We test each year independently (2023, 2024, 2025, 2026) to detect alpha decay. 2026 serves as an out-of-sample check on hypotheses developed from 2023 to 2025 data.

## Original Contributions

1. **Asymmetric alpha decay.** Whale buy signals decayed from 2023 to 2026 while sell signals strengthened. We hypothesise this is partly because whale-watching tools broadcast bullish activity more than bearish, but the cause is not definitively established.

2. **Long-horizon whale intelligence.** Whale deposit edge grows monotonically with time horizon: +1.3% at 24h, +4.8% at 1 month, +12.4% at 6 months. The longer-horizon figures carry more overlap between events and deserve more caution than the 24h figure (see Limitations). Whale sellers are not just day-traders, they can have better experience and knowledge and might see structural shifts weeks to months ahead.

3. **Threshold sensitivity, revised.** We initially reported $10M+ deposits during extreme greed achieving a 78.3% hit rate at 24h. Looking closer, the transactions behind that number cluster within a small number of days, which is the same overlapping-observations problem described in the Limitations section, and it is why we do not treat this figure as confirmed (see Discussion).

4. **DeFi dilution hypothesis.** Due to DeFi maturation, withdrawals can mean staking, LP provision, and L2 bridging, which are non-directional. This dilutes the "withdrawal = buy" assumption. However, deposits remain a clean sell signal.

5. **Out-of-sample test.** All hypotheses were developed on 2023 to 2025 data. 2026 data was then fetched separately and tested without any parameter tuning. Deposit signals survive; withdrawal signals do not.

---

## Data Sources

| Source | Data | Records | Period | Cost |
|--------|------|---------|--------|------|
| Dune Analytics | Whale transactions (>$1M) | 646,442 | Jan 2023 to Jul 2026 | Free tier |
| Binance API | Hourly ETH/USDT prices | 30,801 | Jan 2023 to Jul 2026 | Free |
| Binance API | ETH funding rates (8-hourly) | 3,851 | Jan 2023 to Jul 2026 | Free |
| alternative.me | Crypto Fear & Greed Index | 3,075 | Feb 2018 to Jul 2026 | Free |
| Kaggle | Bitcoin news headlines | 5,906 | Jan 2023 to Sep 2024 | Free |
| GitHub (open-source) | Wallet address labels | 52,768 | Snapshot | Free |

## Methodology

### Data Pipeline (Phase 1)
- 646,442 large ETH transactions via Dune Analytics SQL query
- 52,768 wallet addresses labelled by merging [brianleect/etherscan-labels](https://github.com/brianleect/etherscan-labels) and [dawsbot/eth-labels](https://github.com/dawsbot/eth-labels)
- Label coverage: 62.8% of transactions have at least one identified address
- MEV bot candidates flagged for sensitivity analysis

### Transaction Classification (Phase 2)
- Rule-based labelling for known wallets (exchange deposit, withdrawal, DeFi, wallet-to-wallet)
- Initially built an ML classifier (Random Forest) to predict categories for unknown wallets, achieving 71% accuracy on a time-based hold-out. After expanding the label dataset from 30 to 52,768 addresses, label coverage reached 62.8%, reducing reliance on the classifier. It is still used for the remaining ~37% of transactions where both sender and receiver are unknown

Category distribution:

| Category | Count | Share | Meaning |
|----------|-------|-------|---------|
| wallet_to_wallet | 321,257 | 49.7% | Both addresses are non-exchange, non-DeFi. Could be OTC trades, cold storage transfers, or personal wallet shuffling. No directional signal because intent is unknown, so these are excluded from the event study |
| exchange_deposit | 181,105 | 28.0% | ETH sent TO a known exchange. Interpreted as a sell signal. The main reason to deposit to an exchange is to sell |
| exchange_withdrawal | 124,772 | 19.3% | ETH withdrawn FROM a known exchange. Historically interpreted as a buy signal (accumulating), though this assumption has weakened as DeFi matured |
| defi_interaction | 19,308 | 3.0% | One address is a known DeFi protocol (Uniswap, Aave, etc.). Treated as deploying capital (bullish). Shows a slight edge (52% hit rate) but small sample |

### Sentiment Data (Phase 3)
- **Fear & Greed Index:** Daily composite 0-100 (volatility, volume, social media, BTC dominance)
- **Binance funding rate:** 8-hourly real-money sentiment. Positive = longs pay shorts (bullish crowd). Negative = shorts pay longs (bearish crowd)
- **News sentiment:** 5,906 Bitcoin articles scored with VADER. Tested and discarded as uninformative for predicting price. Market-derived sentiment (FnG, funding rate) is far more useful. News sentiment code remains in the repository but is not used in the core analysis
- **Does whale behaviour track the news?** A separate check, over the roughly 20 months where news coverage overlaps the whale data (Jan 2023 to Sep 2024, 484 days with at least one article): daily news sentiment shows no relationship with daily whale net flow (deposit $ minus withdrawal $). Correlation is essentially zero (r=+0.06, not significant), and bearish news days versus bullish news days show near-identical average whale activity (p=0.95). Whales are neither confirming nor contradicting the public narrative; they appear to act independently of it. A second null result for news sentiment, on a different question than the price-prediction test above (see `scripts/run_sentiment_whale_consistency.py`)

### Event Study (Phase 4 - Core Analysis)
- For each whale transaction, compute forward ETH return at 1h, 6h, 24h, 3d, 1w, 2w, 1m, 3m, 6m
- Measure hit rate: did the whale's action correctly predict price direction?
- Compare to **base rate** (all hours under the same market condition) to isolate whale-specific edge
- Yearly stability analysis: 2023, 2024, 2025 analysed independently; 2026 as out-of-sample
- Threshold sensitivity: $1M, $2M, $5M, $10M minimum transaction size
- Statistical significance via binomial test

### Variables Tested

| Dimension | Values |
|-----------|--------|
| **Time horizons** | 1h, 6h, 24h, 3 days, 1 week, 2 weeks, 1 month, 3 months, 6 months |
| **Transaction thresholds** | $1M+, $2M+, $5M+, $10M+ |
| **Years** | 2023, 2024, 2025, 2026 (independent yearly samples; 2026 out-of-sample) |
| **Sentiment regimes** | Extreme fear (FnG <= 25), fear (25 to 45), neutral (45 to 55), greed (55 to 75), extreme greed (>75), negative funding, positive funding |
| **Transaction categories** | Exchange deposit, exchange withdrawal, DeFi interaction |

---

## Results

### 1. Unconditional Hit Rates

Full dataset, 646,442 transactions. Hit rate > 50% = smart money, = 50% = random.

| Action | Direction tested | 1h | 6h | 24h |
|--------|-----------------|-----|-----|------|
| Exchange deposit | Price dropped? | 49.5% | 49.3% | **50.5%*** |
| Exchange withdrawal | Price rose? | **50.5%*** | 50.3% | 49.7%* |
| DeFi interaction | Price rose? | **50.9%*** | **51.1%*** | **52.0%*** |

\* p < 0.05, \*\* p < 0.01, \*\*\* p < 0.001

On average, whales are no better than a coin flip. The edge is conditional.

---

### 2. Conditioned on Sentiment Regime

All results at 24h horizon, full dataset ($1M+ threshold).

#### Whale Withdrawals (buy signal), 24h

| Condition | N | Hit Rate | p-value | Verdict |
|-----------|---|----------|---------|---------|
| Negative funding rate | 16,301 | **55.1%** | < 0.001 | Smart (but decaying) |
| Extreme fear (FnG <= 25) | 19,208 | 50.7% | 0.058 | Random |
| Fear (FnG 25 to 45) | 18,940 | 50.7% | 0.073 | Random |
| Neutral (FnG 45 to 55) | 21,479 | 47.4% | < 0.001 | Wrong |
| Greed (FnG 55 to 75) | 47,830 | **51.4%** | < 0.001 | Slightly smart |
| Extreme greed (FnG > 75) | 17,226 | 45.6% | < 0.001 | Wrong |
| Positive funding | 108,382 | 48.9% | < 0.001 | Wrong |

#### Whale Deposits (sell signal), 24h

| Condition | N | Hit Rate | p-value | Verdict |
|-----------|---|----------|---------|---------|
| Extreme greed (FnG > 75) | 22,038 | **54.4%** | < 0.001 | Smart |
| Neutral (FnG 45 to 55) | 31,054 | **53.9%** | < 0.001 | Smart |
| Fear (FnG 25 to 45) | 29,189 | **50.8%** | 0.010 | Slightly smart |
| Extreme fear (FnG <= 25) | 29,203 | 49.7% | 0.380 | Random |
| Greed (FnG 55 to 75) | 69,474 | 47.8% | < 0.001 | Wrong |
| Negative funding rate | 24,531 | 45.7% | < 0.001 | Wrong |
| Positive funding | 156,427 | **51.2%** | < 0.001 | Slightly smart |

---

### 3. Yearly Stability Analysis

Each year analysed independently with no data leakage between years. 2026 serves as an out-of-sample check: all hypotheses were developed on 2023 to 2025 data, then 2026 was fetched and tested without any parameter changes. Base rate = what any random hour produces under the same market condition. Edge = whale hit rate minus base rate.

#### Whale Withdrawals (buy signal), conditional: during negative funding, 24h

| Year | N | Hit Rate | Base Rate | Whale Edge | Verdict |
|------|---|----------|-----------|------------|---------|
| 2023 | 1,609 | 68.7% | 64.0% | **+4.7%** | Smart |
| 2024 | 1,348 | 72.2% | 62.1% | **+10.1%** | Smart |
| 2025 | 6,467 | 52.3% | 52.3% | +0.1% | Random |
| 2026 | 6,877 | 51.3% | 55.2% | -3.9% | Gone |

#### Whale Withdrawals (buy signal), unconditional, 24h

| Year | N | Hit Rate | Base Rate | Whale Edge | Verdict |
|------|---|----------|-----------|------------|---------|
| 2023 | 18,955 | 50.7% | 50.3% | +0.4% | Slight |
| 2024 | 42,976 | 51.9% | 52.9% | -1.0% | Random |
| 2025 | 46,745 | 48.9% | 50.7% | -1.7% | Random |
| 2026 | 16,007 | 44.9% | 48.3% | -3.3% | Wrong |

#### Whale Withdrawals (buy signal), conditional: during extreme fear, 24h

| Year | N | Hit Rate | Base Rate | Whale Edge | Verdict |
|------|---|----------|-----------|------------|---------|
| 2023 | 58 | 82.8% | 76.4% | +6.4% | Smart |
| 2024 | 671 | 70.5% | 74.5% | -4.0% | Wrong |
| 2025 | 7,509 | 52.2% | 54.2% | -2.0% | Wrong |
| 2026 | 10,970 | 48.3% | 51.3% | -3.0% | Wrong |

#### Whale Withdrawals (buy signal), conditional: during extreme greed, 24h

| Year | N | Hit Rate | Base Rate | Whale Edge | Verdict |
|------|---|----------|-----------|------------|---------|
| 2024 | 15,596 | 47.3% | 49.9% | -2.6% | Wrong |
| 2025 | 1,630 | 29.7% | 34.2% | -4.5% | Wrong |

#### Whale Deposits (sell signal), unconditional, 24h

| Year | N | Hit Rate | Base Rate | Whale Edge | Verdict |
|------|---|----------|-----------|------------|---------|
| 2023 | 26,492 | 49.5% | 49.7% | -0.2% | Random |
| 2024 | 58,215 | 48.0% | 47.1% | +0.9% | Slight |
| 2025 | 71,815 | 51.0% | 49.3% | **+1.7%** | Smart |
| 2026 | 24,436 | 55.6% | 51.7% | **+3.9%** | Smart |

#### Whale Deposits (sell signal), conditional: during negative funding, 24h

| Year | N | Hit Rate | Base Rate | Whale Edge | Verdict |
|------|---|----------|-----------|------------|---------|
| 2023 | 2,194 | 36.2% | 36.0% | +0.2% | Slight |
| 2024 | 2,219 | 31.9% | 37.9% | -6.1% | Wrong |
| 2025 | 9,928 | 47.5% | 47.7% | -0.2% | Random |
| 2026 | 10,190 | 48.9% | 44.8% | **+4.1%** | Smart |

#### Whale Deposits (sell signal), conditional: during extreme greed, 24h

| Year | N | Hit Rate | Base Rate | Whale Edge | Verdict |
|------|---|----------|-----------|------------|---------|
| 2024 | 19,641 | 52.4% | 50.1% | **+2.2%** | Smart |
| 2025 | 2,397 | 71.3% | 65.8% | **+5.5%** | Smart |

#### Whale Deposits (sell signal), conditional: during extreme fear, 24h

| Year | N | Hit Rate | Base Rate | Whale Edge | Verdict |
|------|---|----------|-----------|------------|---------|
| 2023 | 149 | 18.8% | 23.6% | -4.8% | Wrong |
| 2024 | 1,017 | 30.1% | 25.5% | +4.6% | Smart |
| 2025 | 11,477 | 48.7% | 45.8% | **+2.9%** | Smart |
| 2026 | 16,560 | 52.0% | 48.7% | **+3.3%** | Smart |

---

### 4. Threshold Sensitivity

Does the edge scale with transaction size? Higher thresholds = larger, presumably more informed actors.

Note: as ETH price rose from ~$1,200 (Jan 2023) to ~$4,000+ (2026), a fixed $1M threshold represents progressively fewer ETH and a smaller commitment. A $1M transaction in 2023 was ~833 ETH; in 2026 it is ~250 ETH. This means the $1M pool is diluted over time with less committed actors, making higher thresholds increasingly important for isolating genuinely large players.

#### Whale Withdrawals (buy signal) , conditional: negative funding, 24h

| Threshold | 2023 Edge | 2024 Edge | 2025 Edge | 2026 Edge |
|-----------|-----------|-----------|-----------|-----------|
| $1M+ | +4.7% | +10.1% | +0.1% | -3.9% |
| $2M+ | +6.2% | +11.1% | -1.2% | -4.5% |
| $5M+ | +5.4% | +8.1% | -1.5% | -5.1% |
| $10M+ | +5.4% | +15.5% | +0.4% | -5.4% |

Higher thresholds do not recover the withdrawal edge in 2025 or 2026. The signal is gone at all sizes.

#### Whale Deposits (sell signal), unconditional, 24h

| Threshold | 2023 Edge | 2024 Edge | 2025 Edge | 2026 Edge |
|-----------|-----------|-----------|-----------|-----------|
| $1M+ | -0.2% | +0.9% | +1.7% | **+3.9%** |
| $2M+ | -0.5% | +1.1% | +1.8% | +3.2% |
| $5M+ | -1.0% | +0.5% | +1.6% | **+3.4%** |
| $10M+ | -0.4% | +0.9% | +1.9% | **+2.9%** |

Deposit edge is stable across thresholds and growing over time.

#### Whale Deposits (sell signal), conditional: extreme greed (FnG > 75), 24h

| Threshold | 2024 N | 2024 Edge | 2025 N | 2025 Edge |
|-----------|--------|-----------|--------|-----------|
| $1M+ | 19,641 | +2.2% | 2,397 | +5.5% |
| $2M+ | 11,854 | +2.0% | 1,562 | **+8.0%** |
| $5M+ | 5,256 | +1.8% | 752 | **+10.2%** |
| **$10M+** | **2,856** | **+2.4%** | **350** | **+12.5%** |

Edge scales dramatically with transaction size during extreme greed. $10M+ depositors are the most informed actors in the dataset.

---

### 5. Long-Horizon Analysis (1h to 6 Months)

Full dataset. Do whales think in hours, days, or months?

#### Whale Deposits (sell signal), unconditional

| Horizon | N | Hit Rate | Base Rate | Whale Edge | Mean Return |
|---------|---|----------|-----------|------------|-------------|
| 1h | 180,963 | 49.5% | 49.1% | +0.4% | +0.00% |
| 6h | 180,963 | 49.3% | 48.9% | +0.4% | +0.01% |
| 24h | 180,963 | 50.5% | 49.1% | **+1.3%** | +0.05% |
| 3 days | 180,880 | 49.9% | 48.5% | **+1.4%** | +0.09% |
| 1 week | 180,483 | 50.8% | 49.2% | **+1.6%** | +0.14% |
| 2 weeks | 179,833 | 52.9% | 51.1% | **+1.8%** | +0.12% |
| 1 month | 177,925 | 56.0% | 51.2% | **+4.8%** | +0.34% |
| 3 months | 170,165 | 55.0% | 46.9% | **+8.0%** | +1.16% |
| 6 months | 157,120 | 58.5% | 46.1% | **+12.4%** | +1.56% |

The edge grows monotonically with horizon. Whale sellers see structural shifts months ahead.

#### Whale Deposits (sell signal), conditional: during extreme greed (FnG > 75)

| Horizon | N | Hit Rate | Base Rate | Whale Edge | Mean Return |
|---------|---|----------|-----------|------------|-------------|
| 24h | 22,038 | **54.4%** | 51.8% | **+2.7%** | -0.28% |
| 1 week | 22,038 | 48.8% | 49.0% | -0.1% | -0.02% |
| 1 month | 22,038 | **71.3%** | 69.6% | +1.7% | -3.54% |
| 3 months | 22,038 | **81.2%** | 77.2% | **+4.0%** | -18.39% |
| 6 months | 22,038 | **87.5%** | 86.8% | +0.7% | -22.62% |

At 3-6 months, the base rate does most of the work (markets crash after extreme greed regardless). But at 24h and 3 months, whales show edge above the base rate.

#### Whale Withdrawals (buy signal), unconditional

| Horizon | N | Hit Rate | Base Rate | Whale Edge | Mean Return |
|---------|---|----------|-----------|------------|-------------|
| 1h | 124,684 | 50.5% | 50.9% | -0.3% | +0.00% |
| 6h | 124,684 | 50.3% | 51.0% | -0.7% | +0.01% |
| 24h | 124,684 | 49.7% | 50.9% | -1.2% | +0.05% |
| 3 days | 124,620 | 50.2% | 51.5% | -1.3% | +0.07% |
| 1 week | 124,352 | 48.6% | 50.8% | **-2.1%** | +0.06% |
| 2 weeks | 123,946 | 46.6% | 48.9% | **-2.3%** | -0.06% |
| 1 month | 122,593 | 42.7% | 48.8% | **-6.1%** | -0.21% |
| 3 months | 118,061 | 43.9% | 53.1% | **-9.1%** | +0.30% |
| 6 months | 109,087 | 40.9% | 53.9% | **-12.9%** | -0.15% |

Withdrawals become increasingly WRONG at longer horizons. The "withdrawal = buy" assumption is fundamentally broken.

#### Whale Withdrawals (buy signal), conditional: during negative funding

| Horizon | N | Hit Rate | Base Rate | Whale Edge | Mean Return |
|---------|---|----------|-----------|------------|-------------|
| 24h | 16,301 | **55.1%** | 56.4% | -1.3% | +0.46% |
| 3 days | 16,301 | 52.1% | 56.6% | -4.5% | +0.30% |
| 1 week | 16,301 | 49.7% | 55.1% | **-5.4%** | +0.08% |
| 1 month | 15,389 | 43.6% | 48.7% | **-5.2%** | -0.99% |
| 3 months | 13,670 | 49.4% | 58.6% | **-9.3%** | +3.31% |
| 6 months | 9,441 | 57.1% | 66.9% | **-9.8%** | +20.43% |

Even in the strongest condition (negative funding), whales show no edge above base rate beyond 24h.

#### Whale Withdrawals (buy signal), conditional: during extreme fear (FnG <= 25)

| Horizon | N | Hit Rate | Base Rate | Whale Edge | Mean Return |
|---------|---|----------|-----------|------------|-------------|
| 24h | 19,208 | 50.7% | 53.7% | -3.0% | -0.15% |
| 1 week | 18,874 | 47.7% | 52.9% | **-5.1%** | -1.29% |
| 2 weeks | 18,469 | 46.5% | 53.1% | **-6.6%** | -2.26% |
| 1 month | 17,117 | 43.9% | 54.3% | **-10.4%** | -2.73% |
| 3 months | 15,234 | 36.1% | 33.8% | +2.2% | -9.61% |
| 6 months | 8,407 | 28.7% | 30.2% | -1.5% | +2.08% |

Whales who buy during extreme fear are consistently wrong. At 1 month, they are 10.4% worse than random. They might be buying the wrong dips.

---

### 6. Deposit Edge by Year at Long Horizons

Does the deposit sell signal hold across years at longer horizons?

| Horizon | 2023 Edge | 2024 Edge | 2025 Edge |
|---------|-----------|-----------|-----------|
| 24h | -0.2% | +0.9% | **+1.8%** |
| 3 days | -1.0% | -0.4% | **+2.7%** |
| 1 week | +0.0% | -0.7% | **+3.0%** |
| 2 weeks | -3.3% | -1.0% | **+2.7%** |
| 1 month | -3.8% | **+4.4%** | **+3.6%** |
| 3 months | -2.7% | **+7.0%** | **+6.1%** |
| 6 months | +1.6% | **+7.6%** | **+6.9%** |

The deposit edge was absent in 2023 (during bear-to-bull transition, selling was wrong) but emerged in 2024 and strengthened in 2025. At 6 months, whales who deposited in 2024 were +7.6% above base rate; in 2025, +6.9%.

---

### 7. ML Model (Secondary Analysis)

Random Forest with 22 features (whale, sentiment, price momentum) across 24 walk-forward folds:

| Horizon | Baseline | RF Accuracy | Edge |
|---------|----------|-------------|------|
| 1h | 50.9% | 53.3% | +2.4% |
| 6h | 50.5% | 53.3% | +2.9% |
| 24h | 50.9% | 52.9% | +2.0% |

Feature importance shows price momentum and market sentiment dominate; whale features rank near the bottom. This confirms that whale signal is conditional, not standalone, which is consistent with the event study finding.

---

### 8. Drawdown During the Holding Period

The long-horizon results above report only the return at the END of the holding period. A position that finishes +5% may have been -20% at some point along the way, which most traders cannot tolerate even if the signal is "eventually right". This section measures it directly: the maximum adverse excursion (MAE), the worst unrealised loss a trader following the deposit (sell) signal would have marked-to-market before the final outcome, computed from the full hourly price path over the holding window, not just its endpoint.

The table below covers only deposits where the signal was EVENTUALLY correct (price was lower at t+h), but the question is how much pain they required before winning.

#### Whale Deposits (sell signal), unconditional - MAE on eventually-correct trades

| Horizon | N | Mean MAE | Median MAE | P90 MAE | Mean Final Return |
|---------|---|----------|-------------|---------|---------------------|
| 1 week | 91,764 | 2.7% | 2.0% | 6.2% | -6.4% |
| 2 weeks | 95,088 | 4.3% | 3.5% | 9.7% | -8.8% |
| 1 month | 99,620 | 6.0% | 5.2% | 12.9% | -12.8% |
| 3 months | 93,568 | 9.6% | 7.6% | 21.0% | -26.1% |
| 6 months | 92,904 | **20.1%** | 13.3% | **54.4%** | -30.4% |

#### Whale Deposits (sell signal), conditional: extreme greed (FnG > 75) - MAE on eventually-correct trades

| Horizon | N | Mean MAE | Median MAE | P90 MAE | Mean Final Return |
|---------|---|----------|-------------|---------|---------------------|
| 1 week | 10,760 | 2.8% | 2.3% | 6.0% | -8.0% |
| 2 weeks | 11,723 | 4.4% | 3.8% | 10.3% | -9.3% |
| 1 month | 15,707 | 6.3% | 4.9% | 13.7% | -12.0% |
| 3 months | 17,890 | 12.3% | 9.7% | 29.4% | -27.0% |
| 6 months | 19,539 | **15.5%** | 12.4% | 31.7% | -27.6% |

MAE grows with horizon, roughly in step with the edge itself. A 6-month deposit signal that eventually paid off (unconditional case) required tolerating a 20.1% adverse move on average before it did, and the worst 10% of "correct" trades saw the price rise 54.4% against the position first. The edge documented in Section 5 is real, but a trader (or a stop-loss rule) would need to survive drawdowns of this size to actually collect it. Trades where the signal was eventually WRONG are, worse on both counts, e.g. unconditional 6-month misses averaged 80.1% mean MAE (see `results/drawdown_analysis.csv` for the full breakdown).

This does not model an actual stop-loss RULE. MAE tells us the worst point reached, not whether a specific risk-management rule would have survived it. Simulating a concrete stop-loss policy is a next step.

---

## Discussion

### Why did withdrawal edge decay? (evidenced)

In 2023, withdrawing from an exchange mostly meant a bullish view. By 2025, DeFi had matured enough that people withdraw to stake, provide liquidity, bridge to L2s, or interact with protocols. None of those are directional information for price.

The threshold analysis backs this up. If the problem were just small-time actors diluting the signal, $10M+ withdrawals should still show edge. But the signal is dead at every size, which points to a structural change in the meaning of withdrawals.

### Why did deposit edge survive and grow?

The reason to deposit ETH to an exchange hasn't changed. It still means that you want to sell it. So deposits are still a clean sell signal regardless of what is happening in DeFi.

**A stronger case than we initially gave it credit for.** As ETH's price rose from ~$1,200 (2023) to ~$4,000+ (2026), a fixed $1M threshold captured progressively less committed actors: ~833 ETH in 2023 versus ~250 ETH in 2026 (see Limitations). If the deposit edge were simply a composition effect, dilution should have weakened it over time, since the $1M+ pool increasingly contains smaller, less-informed sellers. Instead, the edge grew. The fact that the edge strengthened despite a headwind that should have worked against it shows a genuine and informational advantage.

**Why the edge specifically wasn't arbitraged away is speculative, not evidenced.** One hypothesis is that whale-watching tools (Nansen, Arkham, Whale Alert) disproportionately broadcast buy-flavoured activity because that is what their users want to hear, leaving sell signals less crowded and less arbitraged. We have no direct evidence for this claim. A brief search turned up one indirect, tangential data point: a 2023 study found Bitcoin's price reacts more strongly to Whale Alert tweets around USDT minting events, which are typically read as bullish.

What we can say more confidently is that the edge scales with transaction size. $10M+ deposits during extreme greed show the strongest signal in the dataset. They are large actors making deliberate moves at market tops.

### Whale sellers think in months

The deposit edge at 24h is small (+1.3%). At 1 month it is +4.8%. At 6 months, +12.4%. These whales are not day-trading. They are making structural calls about where ETH is heading over the next quarter or two, and they are right more often than random. The mean return after whale deposits during extreme greed is -22.6% at 6 months. The price crashes, and the whales were already out.

### What happened to the 78.3% claim

We initially reported that $10M+ deposits during extreme greed correctly predicted a 24h price drop 78.3% of the time. Looking more closely at where that number actually came from, the transactions behind it are concentrated in a small number of days, all within a few weeks of 2025, not spread across the dataset. When the events behind a statistic cluster together like that, they are largely describing the same handful of market moves rather than many separate, independent ones. That is the overlapping-observations problem described in the Limitations section, and it is exactly why we do not trust this specific number, even though the underlying transactions and the arithmetic behind it are entirely real.

---

## Limitations

1. **Long-horizon observations overlap.** At 1-month and longer horizons, thousands of whale events are measuring the same price move. The binomial p-values overstate significance. The edge column (whale vs base rate) is more informative than raw hit rates or p-values at these horizons. See Discussion for a worked example of this problem breaking a specific headline claim.

2. **DeFi dilution is untested.** We hypothesise that withdrawal edge decayed because withdrawals increasingly represent non-directional DeFi activity. Testing would require tracing post-withdrawal activity on-chain (e.g. did the ETH go to a staking contract or a cold wallet?).

3. **On-chain latency.** Whale transactions are visible after block confirmation (~12 seconds), but monitoring, processing, and executing a response trade adds delay.

4. **Fixed USD threshold ignores ETH price growth.** A $1M transaction was ~833 ETH in 2023 but only ~250 ETH in 2026. The $1M pool gets diluted with smaller actors over time. An ETH-denominated threshold or inflation-adjusted threshold would be more rigorous but harder to compare across years. 

5. **Deposits are not screened for round-tripping.** The "whale sellers think in months" framing assumes a deposit reflects a deliberate long-horizon view, but we do not know why a whale deposited. Checking whether the same address later receives a withdrawal: 24.1% of deposits see the same address withdraw within 24 hours, 29.1% within a week, which is inconsistent with a persistent multi-month directional view for at least a meaningful minority of events. This checks only whether a subsequent withdrawal occurred, not whether the amount matches the deposit, so it cannot distinguish genuine short-term round-tripping from unrelated later activity through the same address. The aggregate statistical pattern (deposits preceding declines) may still hold regardless of individual intent, but the narrative that whales are "making deliberate calls" should be read as a description of the aggregate, not a claim about every individual transaction.

---

## Interactive Dashboard

Live: https://crypto-whale-signals-and-sentiment-lkhygb3594bbrogn23qbps.streamlit.app/

To run it locally instead:

```bash
# Rebuild the pre-computed data (only needed after the underlying data changes)
python scripts/build_dashboard_data.py

# Launch the dashboard
streamlit run app/dashboard.py
```

Seven sections: deposit edge by horizon (1h to 6m), yearly stability (alpha
decay, horizon-selectable), threshold sensitivity (horizon-selectable),
sentiment-conditioned hit rates, a return-distribution regime explorer,
the deposit-vs-withdrawal asymmetry by year (horizon-selectable), and
limitations.

The dashboard does not read the raw dataset at runtime. `scripts/build_dashboard_data.py`
pre-computes every number it shows into `app/dashboard_data.json` (~480 KB),
and `app/dashboard.py` reads only that file. This is what lets it deploy to
Streamlit Community Cloud without the 187 MB raw whale dataset, which is
gitignored and never committed.

## Repository Structure

```
whale_signals/
├── app/
│   ├── dashboard.py             # Streamlit dashboard (reads dashboard_data.json only)
│   ├── dashboard_data.json      # Pre-computed dashboard numbers (~480 KB, committed)
│   └── requirements.txt         # Minimal deps for the Streamlit Cloud build
├── .streamlit/
│   └── config.toml              # Dark quant-terminal theme
├── data/
│   ├── raw/                    # Raw data (gitignored)
│   ├── processed/              # Cleaned datasets (gitignored)
│   └── reference/              # 52,768 wallet labels (version-controlled)
├── src/
│   ├── data/                   # Dune client, price fetcher, sentiment fetcher
│   ├── features/               # Feature engineering, Phase 4 feature matrix
│   ├── models/                 # Transaction classifier, price predictor
│   ├── sentiment/              # VADER scorer, hourly aggregator
│   └── analysis/               # Event study (hit rates, walk-forward, threshold)
├── scripts/
│   ├── run_event_study.py      # Core hit rate analysis
│   ├── run_walk_forward.py     # Year-by-year walk-forward validation
│   ├── run_threshold_sensitivity.py  # Transaction size analysis
│   ├── run_phase4_features.py  # Feature matrix builder
│   ├── run_phase4_model.py     # ML walk-forward model
│   ├── run_sentiment_pipeline.py  # News sentiment scorer
│   ├── run_drawdown_analysis.py # Maximum adverse excursion (Section 8)
│   ├── run_sentiment_whale_consistency.py # News sentiment vs whale activity
│   └── build_dashboard_data.py # Pre-computes app/dashboard_data.json
├── tests/                      # Unit tests
├── docs/                       # Design notes, project arc
└── results/                    # Walk-forward, threshold, and drawdown CSVs, charts
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Core analysis: event study hit rates
python scripts/run_event_study.py

# Walk-forward validation by year
python scripts/run_walk_forward.py

# Threshold sensitivity analysis
python scripts/run_threshold_sensitivity.py

# ML model (walk-forward Random Forest)
python scripts/run_phase4_model.py

# Maximum adverse excursion (drawdown during the holding period)
python scripts/run_drawdown_analysis.py

# Does whale activity track news sentiment?
python scripts/run_sentiment_whale_consistency.py

# Rebuild the dashboard's pre-computed data, then launch it
python scripts/build_dashboard_data.py
streamlit run app/dashboard.py
```

Note: Raw whale data requires Dune Analytics API access. Pre-processed data is not included due to file size.
