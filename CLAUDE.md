# Claude Code Instructions for `whale_signals`

This file is read by Claude Code on every interaction. It defines how Claude should approach this codebase.

## Project Context

This is a research project combining on-chain Ethereum whale tracking with NLP sentiment analysis to model short-term price impact. The goal is to test whether large on-chain transactions (>$1M equivalent) systematically precede price movements in ETH, and whether sentiment moderates this relationship.

The author (Fred) is a Tonmeister (audio engineering / DSP) student applying to MSc FinTech and Quantitative Finance programmes — particularly HKU's MFFinTech, which has strong blockchain and crypto emphasis. The project must demonstrate:

1. Genuine engagement with blockchain technology, not just "AI applied to crypto prices"
2. Deep understanding of every design decision, not just working code
3. Rigorous methodology (no look-ahead bias, proper walk-forward validation)
4. Honest reporting of results, including negative findings
5. Clean, modular, well-documented code suitable for showing to admissions panels

## Communication Style

- **Be direct and honest.** Never sugarcoat. If a design choice is questionable, say so before coding.
- **Explain the why, not just the what.** Every function comes with reasoning for design decisions.
- **Use analogies to audio/DSP where helpful.** Fred's background is signal processing; concepts like filtering, noise, event detection, sampling rates, and adaptive systems translate naturally to finance.
- **Default to British English** in comments and documentation.
- **No emojis** in code, comments, or commit messages.

## Code Standards

### Commenting

Fred is self-taught in pandas/sklearn/scipy/web3.py. Do not assume familiarity with library methods. Every non-trivial method call should have a brief inline comment explaining what it does.

Example:
```python
# pandas .merge with on= joins two DataFrames on a shared column
# how='left' keeps all rows from the left DataFrame, adds NaN where right is missing
merged = whale_txs.merge(prices, on='hour', how='left')
```

### Structure

- **Modular design.** Every component (data loader, classifier, sentiment scorer) is independent and testable.
- **No hardcoded paths.** Use `config.py` or environment variables.
- **No look-ahead bias.** Any time we use future data to compute features used for prediction, it must be explicitly flagged and the function marked `# ACADEMIC USE ONLY` or similar.
- **Type hints on all functions.** Use `typing` module.
- **Docstrings on all public functions.** Numpy or Google style.

### Testing

- Every data transformation needs a unit test covering edge cases (empty data, missing values, timezone issues).
- Walk-forward backtests must be tested with synthetic data first to confirm no look-ahead is occurring.

## Project Phases

Build in this order. Do not jump ahead without completing the previous phase.

### Phase 1: Whale Data Pipeline (Weeks 1-3)
- Etherscan API client for transaction history
- Web3.py for direct chain queries when needed
- Filter transactions by USD-equivalent value (use ETH/USD price at transaction time)
- Build labelled-address database from Etherscan public labels (exchanges, DeFi protocols)
- Output: clean CSV of whale transactions with source/destination labels

### Phase 2: Wallet & Transaction Classification (Weeks 4-5)
- Feature engineering: wallet labels, transaction size, gas price, sender history, time features
- Train Random Forest classifier to categorise transactions:
  - Exchange deposit (selling pressure)
  - Exchange withdrawal (accumulation)
  - DeFi interaction
  - Wallet-to-wallet
- Validate on hold-out set; analyse confusion matrix and feature importance

### Phase 3: Sentiment Pipeline (Weeks 6-7)
- Reddit fetcher via PRAW (r/CryptoCurrency, r/Bitcoin, r/Ethereum)
- CryptoPanic news API fetcher
- VADER scoring of titles and bodies
- Hourly aggregation: mean sentiment, post volume, polarity ratio

### Phase 4: Price Impact Prediction (Weeks 8-10)
- Combine whale signals + sentiment + price features
- Walk-forward training: at each time step, model trained only on prior data
- Predict directional price movement at t+24h
- Compare logistic regression, Random Forest, XGBoost
- Honest evaluation: edge over random baseline, not raw accuracy

### Phase 5: Evaluation and Backtest (Weeks 10-12)
- Realistic P&L simulation with transaction costs
- Sensitivity analysis: which whale categories produce strongest signal?
- Sentiment as moderator: does it improve signal in certain regimes?
- Final findings document in `docs/findings.md`

### Phase 6: Documentation and Polish (Weeks 12-13)
- Final README with methodology and findings
- Charts in `results/charts/`
- Methodology defence in `docs/methodology.md`

## Anti-Patterns to Avoid

- **Don't generate large blocks of code without explaining the reasoning.** Fred needs to understand and defend every decision in interviews.
- **Don't use unfamiliar libraries without justification.** If switching tools, explain why.
- **Don't add features without testing them.** Build incrementally.
- **Don't fake or fabricate results.** If a model performs at chance level, report it honestly.
- **Don't paper over look-ahead bias.** If you spot it, flag it loudly.
- **Don't treat the project as "GARCH with sentiment on crypto prices".** The blockchain dimension — actual on-chain transaction analysis — is the differentiator. Keep it central.

## Honest Defaults

- If asked to implement something Fred doesn't fully understand, **stop and explain the concept first** before writing code.
- If a proposed approach has known issues (e.g. on-chain whale data is noisy, classification labels are imperfect), **say so before coding**.
- If a library does something non-obvious, **explain what it's doing under the hood** before using it.

## Domain Notes

A few facts Fred should internalise as the project progresses:

- **Exchange wallets are publicly known.** Etherscan labels addresses owned by Binance, Coinbase, Kraken, etc. This is how whale-watching firms classify movements.
- **Whale-to-exchange ≠ guaranteed selling.** Some users park funds at exchanges for various reasons. The classifier should output probabilities, not certainties.
- **Gas price as urgency signal.** When a whale pays high gas, they want fast confirmation — often a stronger signal.
- **MEV bots and arbitrage flows** appear as "whale" transactions but are not directional bets. Filtering these out matters.
- **The 24-hour prediction horizon is arbitrary.** Test multiple horizons (1h, 6h, 24h, 72h) to find where the signal lives.

## Output Expectations for Each Session

When making changes, always:
1. Explain what you're about to do and why.
2. Show the code with inline comments.
3. Note any caveats, limitations, or things Fred should verify.
4. Suggest the next logical step.
