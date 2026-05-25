# On-Chain Whale Behaviour and Sentiment-Adjusted Price Impact

A research project combining on-chain Ethereum whale tracking with NLP sentiment analysis to model short-term price impact of large transactions.

## Research Question

Do large on-chain transactions (>$1M equivalent) systematically precede price movements in ETH, and does social media sentiment moderate this relationship?

## Why This Matters

In traditional markets, institutional positioning is reported quarterly with significant lag. On a public blockchain like Ethereum, every transaction is visible in real-time. This creates an opportunity to:

1. Identify large holders' behaviour as it happens
2. Classify their intent (selling pressure vs accumulation)
3. Test whether their actions predict price movements

This is genuinely impossible to do with stocks or commodities. It is a uniquely crypto-native research question.

## Methodology

### 1. Whale Data Pipeline
- Extract large ETH transactions (>1000 ETH or >$1M USD-equivalent) via Etherscan API and Web3.py
- Classify wallets using Etherscan's labelled address database:
  - Centralised exchange addresses
  - Known DeFi protocol contracts
  - Other (private wallets)

### 2. Transaction Classification (ML)
- Build a supervised classifier (Random Forest / XGBoost) to categorise whale movements:
  - **Exchange deposit** — potential selling pressure
  - **Exchange withdrawal** — potential accumulation
  - **DeFi interaction** — yield-seeking or hedging
  - **Wallet-to-wallet** — neutral or internal transfer
- Features: source/destination labels, transaction size, gas price, time of day, sender age and history

### 3. Sentiment Layer
- VADER sentiment scoring of Reddit (r/CryptoCurrency, r/Bitcoin, r/Ethereum) and crypto news headlines
- Hourly aggregation into mean sentiment scores
- Optional: FinBERT for financial-domain sentiment

### 4. Price Impact Prediction (ML)
- Combine whale classification + sentiment + price features
- Train classification model: given a whale signal at time t, predict directional price movement at t+24h
- Walk-forward validation to prevent look-ahead bias

### 5. Evaluation
- Out-of-sample classification accuracy and edge over random baseline
- Profit-and-loss simulation of acting on signals (with realistic transaction costs)
- Honest reporting: where the model fails and why

## Repository Structure

```
whale_signals/
├── data/
│   ├── raw/                # Raw API responses (gitignored)
│   └── processed/          # Cleaned datasets
├── src/
│   ├── data/               # Etherscan, Web3, Reddit, CryptoPanic fetchers
│   ├── classification/     # Wallet and transaction classifiers
│   ├── sentiment/          # VADER and optional FinBERT scorers
│   ├── prediction/         # Price impact prediction models
│   └── evaluation/         # Walk-forward backtest and metrics
├── notebooks/              # Exploratory analysis
├── tests/                  # Unit tests
├── results/                # Output charts, metrics, model artefacts
└── docs/                   # Methodology and design notes
```

## Status

In active development. Findings will be documented in `docs/findings.md`.

## Author

Fred Zheng — BSc Music and Sound Recording (Tonmeister), University of Surrey.

Applying signal processing methods from audio engineering — time-series filtering, adaptive systems, event detection — to financial market data.
