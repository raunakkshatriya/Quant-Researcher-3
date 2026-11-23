# Quant-Researcher-3

Different trading strategies, built one small step at a time and tested on real historical data.

## Rules of this project
* **Train period:** from the first available date up to **2025-12-31**. We look at this data to choose settings.
* **Test period:** **2026-01-01** onwards. We only look at it after the settings are fixed. This is the honest check.
* Every strategy starts with **$100** and the dashboard shows what it would be worth.
* The test period is **locked**: scripts show train results only; add `--test` for one logged look. See `docs/INDUSTRY_READINESS.md` for the standards every step follows.
* Every strategy gets its own folder with a `description.txt` explaining what it does, how, and which stocks/indexes it uses.
* This is research and paper-trading practice, not investment advice.

## Run the tests
```
python -m pytest -q
```

## Roadmap
| Step | What | Status |
|---|---|---|
| 0 | Set up Python, download prices | done when you see "Step 0 passed" |
| 1 | Tiny backtest engine (positions -> $100 curve) | |
| 2 | Sector rotation | `python step2_sector_rotation.py` |
| 3 | Cross-sectional momentum (+ research lab: trial log, deflated Sharpe, walk-forward, locked test) | `python step3_momentum.py` |
| 10 | News sentiment: Alpaca/Benzinga headlines scored by FinBERT on GPU (step10_news_sentiment.py; setup in docs/ALPACA_SETUP.md) |
| 9b | Re-score sector rotation and trend on the current Sharpe scale vs fair benchmarks (step9_rescore_on_current_scale.py) |
| 3b | Re-check momentum vs equal-weight buy & hold of the same 50 stocks (alpha/beta, train only) |
| 4 | Cross-asset trend following | `python step4_trend_riskparity.py` |
| 5 | Pairs trading: 49 stocks vs 11 sector ETFs, compared (crypto trend postponed) | `python step5_pairs.py` |
| 6 | FX carry + momentum (9 currencies vs USD) | `python step6_fx_carry.py` |
| 7 | SMA trend filter, the baseline (49 stocks) | `python step7_sma.py` |
| 8 | Execution research, Stage A: capacity study (cost vs account size) | `python step8_capacity.py` |
| 9 | Crypto trend (postponed) | |
| 10 | News sentiment | |
| 11 | Dashboard | |

## Setup (once)
```bash
git clone https://github.com/raunakkshatriya/Quant-Researcher-3.git
cd Quant-Researcher-3
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Step 0
```bash
python step0_check_data.py
```
You should see a table of Apple prices and, at the end, `Step 0 passed`.

## Each time I push a new step
```bash
git pull
pip install -r requirements.txt   # only matters if I added a package
```
Then run the command I give you for that step.
