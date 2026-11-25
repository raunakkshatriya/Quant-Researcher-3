# Daily monitor

- Last final price bar used: **2026-11-24** (job ran 2026-11-25 05:03 UTC)
- Paper tracking: every book = $100 at the close of 2026-11-20, then followed forward. Books start already holding their positions (no entry cost), like the backtests.
- Status: **OK**

## Books

| Book | As of | Paper NAV | Positions | Invested | Turnover vs previous log |
|---|---|---|---|---|---|
| Momentum | 2026-11-24 | $104.26 | 10 | 100% | 0.0% |
| Trend / risk parity | 2026-11-24 | $102.17 | 3 | 87% | 0.0% |
| SMA trend filter | 2026-11-24 | $101.71 | 23 | 47% | 3.1% |
| FX carry | 2026-11-24 | $99.81 | 6 | 0% | 0.0% |

## Target weights for the next trading day (top 10 per book)

- **Momentum**: MU 10.0%, AMD 10.0%, INTC 10.0%, LRCX 10.0%, MRK 10.0%, AMAT 10.0%, DELL 10.0%, PANW 10.0%, GEV 10.0%, TXN 10.0%
- **Trend / risk parity**: SPY 34.8%, EFA 31.7%, DBC 20.4%
- **SMA trend filter**: AAPL 2.0%, MSFT 2.0%, AMZN 2.0%, TSLA 2.0%, LLY 2.0%, WMT 2.0%, V 2.0%, JNJ 2.0%, XOM 2.0%, MA 2.0%
  - since previous log: entered none; exited AMD, ANET, TXN
- **FX carry**: GBP 16.7%, JPY -16.7%, CHF -16.7%, AUD 16.7%, NOK 16.7%, SEK -16.7%

## Data notes (not failures)

- NVDA: +31% move on 2026-11-23 (check for a split or data error)

---
This file records what the frozen strategies hold and how $100 would have grown since the paper start. It does not judge them. Reviews are on scheduled dates only (see monitor/README.md).
