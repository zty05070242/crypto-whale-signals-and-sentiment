# Design Notes

A record of design decisions, reasoning, and trade-offs as the project develops. The goal is a clear record Fred can refer to in interviews.

## Why on-chain whale tracking over GARCH-only sentiment?

The original project pitch was GARCH-X with sentiment on BTC/ETH prices. The problem: pulling BTC prices from Binance is no different from pulling stock prices from Yahoo Finance. The fact that the asset is a cryptocurrency doesn't make the project "blockchain."

Whale tracking uses data that **only exists on a blockchain**: every individual transaction, with sender, recipient, value, and gas — all public, all real-time. There is no equivalent in traditional finance. This makes the project genuinely crypto-native.

## Why Ethereum specifically?

Three reasons:

1. **Public labelled addresses.** Etherscan has the most comprehensive labelling of exchanges, DeFi protocols, and known entities. Bitcoin labels are sparser.
2. **Smart contract context.** Many whale transactions interact with DeFi protocols (Aave, Uniswap). This adds analytical depth.
3. **Higher transaction count.** More signal density per unit of time than Bitcoin.

## Why $1M+ threshold for "whale"?

Arbitrary but defensible. Industry analytics firms (Nansen, Glassnode, Arkham) typically use thresholds in the $100k-$10M range depending on context. $1M filters out retail-scale movements while keeping enough events to do statistics on.

The threshold should be a configurable parameter. The honest finding may be that smaller transactions ($100k-$1M) carry more signal because larger transactions are often internal exchange operations.

## The classification problem in detail

Naive view: "exchange deposits = selling, exchange withdrawals = buying"
Reality: substantially noisier.

Cases where exchange deposits are NOT selling:
- Internal exchange wallet rebalancing (Binance moving cold-to-hot)
- Market maker funding (not directional)
- Withdrawal preparation (paradoxically, deposits sometimes precede withdrawals)

Cases where wallet-to-wallet transfers ARE significant:
- Whale moving funds to a known DeFi protocol about to launch a new vault
- Funds flowing into a wallet historically associated with smart accumulation

This is why a classifier with probability outputs is more useful than rigid rules. We never get 100% certainty — we get "this transaction has a 73% probability of being directional selling pressure."

## Walk-forward validation philosophy

The most common mistake in financial ML is look-ahead bias. Prevention:

- Models refit using only data available up to time t before predicting time t+1
- Sentiment aggregation uses only past posts
- Wallet classification labels use only Etherscan labels available before the prediction window
- Hyperparameters tuned on early data only

This is the audio equivalent of a **causal filter** — output at time t depends only on inputs at time ≤ t.

## Data alignment

- Ethereum blocks ~12 seconds apart
- Bucket whale transactions into UTC hourly windows
- Bucket Reddit posts and news into matching hourly windows
- Bucket prices into matching hourly windows from Binance or CoinGecko
- Join everything on the hourly index

Hours with zero whale transactions are common — handled as zeros in feature vector, not NaN. Hours with zero sentiment data get forward-filled with a decay (older sentiment becomes weaker proxy).

## MEV and bot filtering

Maximum Extractable Value bots execute large transactions for arbitrage, sandwich attacks, and liquidations. These look like "whale" transactions but are not directional bets on price.

Two filtering approaches:
1. Identify and exclude known MEV bot addresses (Flashbots, Eden Network)
2. Filter by transaction patterns — MEV bots typically operate within single blocks with cyclical transaction sequences

Imperfect, but reduces noise substantially.

## Honest expectations about results

Whale tracking firms make money selling this exact analysis. If the signal were strong and obvious, large funds would have arbitraged it away.

Realistic expectations:
- **Best case**: 55-60% directional accuracy at 24-hour horizon (8-10% edge over random)
- **Likely case**: 51-54% accuracy, signal only present in subset of transaction types
- **Worst case**: No edge after accounting for transaction costs

Any of these is a valid finding. The honest reporting matters more than the magnitude of the edge.

## Open questions

- What's the right prediction horizon? Test 1h, 6h, 24h, 72h.
- Should we predict direction (binary) or magnitude (regression)?
- Does sentiment moderate the signal or operate independently?
- Is there a regime effect — does whale signal differ in bull vs bear markets?
