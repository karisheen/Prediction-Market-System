# Initial Venue Decision

Decision date: 2026-07-28

## Decision

Use **Kalshi** for the first read-only market-data adapter and historical
backtesting integration. Add **Polymarket** later as a second, read-only
cross-venue signal source. Do not add automated execution for either venue in
the initial system.

## Why Kalshi is first

- Its official REST quick start exposes market metadata and current order books
  without API credentials.
- Official historical endpoints expose archived markets, trades, and
  one-minute/hourly/daily candlesticks, which directly supports walk-forward
  testing.
- It has separate demo and production environments for future authenticated
  testing.
- Current market categories include crypto, finance, commodities, and politics,
  matching the intended expansion path.
- Crypto contracts state their resolution benchmark and rules. The adapter will
  preserve those rules and reject unsupported path-dependent contracts rather
  than silently applying the wrong model.

Kalshi WebSocket sessions require an API key even for public market-data
channels. The first adapter therefore uses unauthenticated REST polling. That is
sufficient for manual-review alerts and avoids requesting credentials early.

## Why Polymarket is second

- Public market discovery, prices, and CLOB order books are available without a
  signer or user credentials.
- It has broad crypto coverage and can provide useful cross-venue comparison.
- Private trading uses wallet signatures plus two layers of CLOB
  authentication.
- Fees are dynamic by market and must be retrieved from current market
  parameters.
- Official geoblocking documentation lists the United States and many other
  jurisdictions as restricted for opening positions. Read-only data remains
  useful, but execution eligibility must be confirmed before treating it as a
  tradable primary venue.

## Important model boundary

The structural engine distinguishes terminal thresholds from touch barriers.
Terminal contracts use the probability of finishing beyond the strike. An
early-close contract is routed to a geometric-Brownian first-passage model only
when its direction metadata and rules explicitly define touch semantics. Ambiguous
early-close rules are rejected rather than silently assigned the wrong model.
Both models preserve the stated benchmark and rule text for resolution-risk review.

## References

- [Kalshi public market data](https://docs.kalshi.com/getting_started/quick_start_market_data)
- [Kalshi order-book semantics](https://docs.kalshi.com/getting_started/orderbook_responses)
- [Kalshi historical data](https://docs.kalshi.com/getting_started/historical_data)
- [Kalshi API environments](https://docs.kalshi.com/getting_started/api_environments)
- [Polymarket integration surfaces](https://docs.polymarket.com/getting-started/api)
- [Polymarket public CLOB methods](https://docs.polymarket.com/trading/clients/public)
- [Polymarket fees](https://docs.polymarket.com/trading/fees)
- [Polymarket geographic restrictions](https://docs.polymarket.com/api-reference/geoblock)
- [Mörters and Peres, *Brownian Motion*](https://people.math.ethz.ch/~grunewal/teaching/FS2015/MortersPeres.pdf)
