# FX Portfolio Monitor

A Streamlit dashboard that monitors the P&L and risk of a spot FX book reported in USD.
**Live:** `<add your Streamlit Community Cloud URL here>` · **Run locally:** `pip install -r requirements.txt && streamlit run app.py`

Every tab has a **📘 How it works** box (click to open). It explains each number in plain English, with its formula,
the exact line of code that computes it, and what each input means. `docs/pm_brief.pdf` is a longer walkthrough
with worked examples from this book and answers to likely questions.

## How the brief is covered

| Requirement | Where in the dashboard |
|---|---|
| Read an existing portfolio from a file (own sample file) | `data/portfolio.csv` (trade date, pair, side, notional, entry price), created from `data/sample_portfolio.csv`. Synced both ways: trades booked in the dashboard are written to the file, and edits to the file appear in the dashboard within ~3 s. |
| Current and historical market prices | Yahoo Finance (`yfinance`): 3 years of daily closes and recent hourly prices for 26 currencies, cleaned and checked (see 1.5) |
| Current portfolio, positions, USD P&L daily and since inception | Overview (headline numbers, exposure, P&L breakdown); Positions & risk (positions table, blotter of every trade) |
| Position VaR, portfolio VaR, marginal / component VaR | Positions & risk: portfolio VaR, ES, diversification and backtest as tiles at the top; per position in the Positions table: position VaR, component VaR, marginal VaR, contribution to ES, % of risk, marginal ES (see 1.3) |
| Additional analytics for a PM | P&L attribution, pre-trade check, rich / cheap, dollar and peer analysis, sessions, seasonality, markets panel, limits and data-quality flags (see 1.4) |

---

## 1. Methodology and major assumptions

### 1.1 Core idea
1. **A spot trade is two currency amounts.** Buying N EURUSD at price P gives **+N EUR** and **−N × P USD**. Its USD
   value changes only when one of those currencies moves against the dollar.
2. **Trades are stored, never positions.** Positions, P&L and risk are rebuilt from the trade file on every refresh,
   so they can never drift from the blotter.
3. **Risk = today's holdings × what markets actually did.** Today's positions are replayed over real past days; no
   bell curve is assumed.

**Notation:** `Q_c` = USD price of one unit of currency c (USD = 1); pair price `S = Q_base / Q_quote`;
`N` = signed notional (+ buy, − sell); `P` = trade price; `E_c` = USD value held in currency c.

### 1.2 Positions and P&L

| Number | In plain English | Formula |
|---|---|---|
| Mark-to-market (per trade) | Value both currency amounts at today's prices | `N·Q_base − N·P·Q_quote` (= N(S−P) for xxxUSD; converted to USD for USDxxx and crosses) |
| Inception P&L | What all live trades are worth today vs their trade prices | Σ mark-to-market now |
| Daily P&L | Change since the last daily close (08:00 SGT); a trade done since then counts in full | Σ [MTM now − MTM at the previous close] |
| Net exposure by currency | USD value of everything owned (+) or owed (−) in each currency, at today's price | `E_c = (Σ amounts of c) × Q_c,now` |
| Avg entry (break-even) | Price at which a pair's total P&L would be zero | `Σ N·P / Σ N` |
| Gross exposure | Total size of all currency bets, longs and shorts | `Σ |E_c|` |

### 1.3 Risk: historical simulation, 1 day, 95%, last 500 completed days
Where it appears: the **Positions & risk** tab shows the portfolio numbers as tiles at the top (VaR, ES,
diversification, backtest) and the per-position numbers as columns in the **Positions** table. Worked examples use
this book: AUDUSD (the biggest risk) and GBPUSD (a hedge). All code is in `risk.py` → `compute_risk`.

**Step 1: replay the last 500 days on today's positions.** For each completed day, work out what each position
would make or lose if that day's currency moves happened again today. Its USD exposure in each currency × that
currency's % move, added up. This gives a table **X** (500 days × positions), and the book's result per day
**P** = the sum across positions: 500 possible "tomorrows".
```python
returns = md.simple_returns(list(pair_exp.columns)).loc[:md.last_complete_date]   # daily % moves, completed days only
return pd.DataFrame(r.values @ exposure[ccys].values.T, index=r.index, columns=exposure.index)   # X[day, position]
p = p_full.iloc[-lookback:]                                                        # P[day], last 500 days
```

**Step 2: portfolio numbers (the tiles).**

| Measure | Question it answers | How it is calculated | Code | This book |
|---|---|---|---|---|
| **Portfolio VaR** | "What loss is exceeded on only 1 day in 20?" | Sort the 500 results; take the 5th percentile (between the 25th and 26th worst), as a positive loss | `var_hs = hs_var(p, conf)` → `-np.percentile(arr, (1 - conf) * 100)` | $44.6k |
| **Expected shortfall (ES)** | "When it is a bad day, how bad on average?" | Average of the 25 worst results (5% of 500) | `tail = p.nsmallest(k).index`; `es = float(-p.loc[tail].mean())` | $69.6k |
| **Diversification** | "How much risk cancels out between positions?" | Sum of each position's own VaR − the book's VaR | `diversification=float(position_var.sum() - var_hs)` | $147.8k − $44.6k = $103.2k |
| **VaR backtest** | "Has VaR been too low?" | For each of the last 250 days, VaR from the 500 days *before* it vs that day's result; a breach = a bigger loss. About 12–13 expected | `var_i = hs_var(vals[i - lookback:i], conf)` | 4 breaches |

**Step 3: per-position numbers (columns in the Positions table).** *USD size* = |net notional × today's USD price of
the first currency| (AUDUSD $2.80m, GBPUSD $2.65m).

| Column | Question it answers | How it is calculated | Code | AUDUSD | GBPUSD |
|---|---|---|---|---|---|
| **Position VaR** | "How much can this position lose **on its own**?" | VaR of the position's own 500 results, as if it were the only position | `position_var = x.apply(lambda col: hs_var(col, conf))` | $23.8k | $19.5k |
| **Component VaR** | "How much of the **book's VaR** is this position?" Adds up to VaR | Its average loss on the 11 days ranked around the VaR day (20th–30th worst), scaled so the column sums to VaR. One day alone would be too noisy | `raw = -x.loc[band].mean()`; `component_var = raw * (var_hs / raw.sum())` | $17.2k | −$4.4k |
| **Marginal VaR / $1m** | "How much would VaR change per $1m more of it?" (a guide, for small changes) | Component VaR ÷ USD size × $1m | `marginal_var = component_var / size.replace(0, np.nan) * 1e6` | $6.1k | −$1.7k |
| **Contribution to ES** | "What does this position do on the **book's 25 worst days**?" Adds up exactly to ES; negative = hedge | Its average P&L on those same 25 days, as a loss | `component = -x.loc[tail].mean()` | $30.1k | −$10.3k |
| **% of risk** | "What share of the tail loss is this?" | Contribution ÷ ES | `pct = component / es` | 43% | −15% |
| **Marginal ES / $1m** | "How much would ES change per $1m more of it?" | Contribution ÷ USD size × $1m | `marginal = component / size.replace(0, np.nan) * 1e6` | $10.8k | −$3.9k |

**Reading the numbers**
- **Biggest risk: AUDUSD.** AUD fell 1.08% on average on the book's 25 worst days → $2.80m × 1.08% ≈ $30.1k = 43% of
  ES. Adding $0.1m more raises ES by $1,078, exactly as marginal ES predicts.
- **Why a hedge has a positive VaR but negative risk columns.** Position VaR asks "how much can it lose on *its own*
  bad days?" (GBPUSD loses when GBP rallies: $19.5k). The other columns ask "what does it do on the *book's* bad
  days?". The book is long AUD, NZD, CAD and EUR, and on its worst days GBP usually fell too (−0.39%), so the
  GBPUSD short **made** money on 19 of the 25 worst days: contribution −$10.3k, −15% of risk. Selling $0.1m more cuts
  ES by $387, as predicted. It is a hedge only for the kind of bad days seen in the last two years.
- **ES columns vs VaR columns.** Both splits add up (to ES and to VaR respectively). The ES split averages 25 days
  and is the main "who uses the risk budget" measure (also the Overview chart). The VaR split rests on fewer days and
  is less stable. VaR jumps from one replayed day to another, so marginal VaR is only a guide (GBPUSD +$0.1m: VaR
  +$56 vs −$165 predicted). For a real trade size, the pre-trade check rebuilds the whole book.

**Why this method and these settings**
- **Historical, not parametric:** FX has fat tails and jumps (SNB 2015, BoJ intervention). Replaying real days keeps
  them at their real size, with no bell-curve assumption.
- **500 days (~2 years):** covers more than one calm year but stays recent (this book's VaR is $35.7k on 250 days vs
  $44.6k on 500).
- **95% and 1 day:** 25 tail days are enough for a stable VaR, ES and backtest (99% would rest on 5 days); the book
  is marked and managed daily.
- **One framework throughout:** VaR, ES and every split come from the same replayed days.

**This book (10 trades, prices to the 9 Oct 2026 close):** VaR $44.6k (89% of the $50k limit), ES $69.6k. AUDUSD is
the biggest risk; GBPUSD and USDCHF are hedges. Every number was checked against an independent recomputation, and
the tabs agree with each other.

### 1.4 Additional analytics (plain-English summary)

| Feature | What it tells the PM | How it is calculated |
|---|---|---|
| **P&L by currency** | Which currencies made or lost money | Each trade's two amounts × (price now − entry price). The entry price is implied by the trade price, so the rows add up exactly to the headline. A pair against USD sits entirely in its currency (NZD row = NZDUSD position). |
| **P&L by session and weekday** | Whether money is made in Asia (08–16 SGT), Europe (16–21) or US (21–08) hours, and on which weekdays | Hourly prices: amounts held × each hour's price change. A trade enters in the hour it was booked. Sessions add up exactly to inception P&L. |
| **Trade ticket and pre-trade check** | What a trade would do before it is booked | The trade is added to a copy of the book and everything is rebuilt: before / after VaR, ES, gross, DXY-equivalent and currency exposure; VaR vs limit; a market-context line; a warning if the price is >5% off market. Confirm writes it to the file. |
| **Rich / cheap** | Whether a currency is unusually strong or weak, and whether a recent move is unusually big | Level z = (today's log price − average of the previous 1M–1Y) ÷ its standard deviation. Move z = move over 1W–1Y ÷ the move a normal period of that length would have. |
| **USD beta and DXY-equivalent** | How much each currency follows the dollar index, and how big a dollar bet the whole book is | DXY rebuilt from the same prices (ICE weights). β = covariance with DXY ÷ DXY variance (3M and 1Y); R² = how consistent. DXY-equivalent = −Σ exposure × β. |
| **Dollar / bloc / own move and peer z** | Whether a move was the dollar, the currency's group (e.g. commodity FX) or the currency itself | Daily log move = yesterday's β × DXY move + median move of its peers (excluding itself) + what is left (own). Peer z = how unusual the own move is vs its history. |
| **Sessions and calendar** | When in the day a currency usually moves, and around which events (holidays, month-end, mid-month, fiscal year-end) | Log moves in fixed SGT windows that add up to the day; z-score vs the previous N days (typed in). Holidays from the `holidays` package. |
| **Seasonality** | What a pair usually does at this time of year | Month-end-to-month-end % returns over the chosen years: average, median, hit rate; year and month paths lined up by day of the year; optional removal of each year's trend (previous 12 months only). |
| **Markets panel** | How the pairs a PM follows moved | Price charts and % change over 1 day, 1 week (5 days) and 1 month (21 days). |
| **Limits, flags and diagnostics** | Whether to act, and whether the numbers can be trusted | VaR limit ($50k) and currency limit ($5m); flags for limit breaches, stale prices, failed checks and backtest breaches; data checks and a searchable log. |

### 1.5 Major assumptions
- **Prices:** Yahoo Finance mid prices (indicative, delayed). Daily close = Yahoo's snapshot at **08:00 SGT** (00:00
  UTC), not the 5pm New York close. "Now" = the latest hourly price.
- **Clock:** a trading day runs 08:00 → 08:00 SGT (computed in UTC, shown in SGT). Weekends don't trade; holidays
  still trade, thinly.
- **Data cleaning:** Yahoo's daily bars are dated one day late and are re-dated; the latest close is rebuilt from
  hourly bars; one-day bad prints (a >6σ move that reverses the next day) are removed; quoted crosses are checked
  against the USD legs.
- **Product:** spot only: no forward points, carry, financing, bid / ask, fees or slippage; trades default to mid.
- **Reporting:** USD. A cross's P&L is converted at today's rate; P&L held in another currency keeps its FX risk.
- **Risk:** 1-day horizon, positions held fixed, every past day equally likely, completed days only (today's
  unfinished day is excluded), no liquidity adjustment.
- **Units:** plain % changes, except log % where parts must add up (sessions, the dollar / bloc / own split).

---

## 2. Limitations and known issues
- **The risk model understates crises.** The book is a **risk-on / carry** book (long AUD, NZD, CAD, EUR; short JPY,
  CHF, GBP), with a −0.89 correlation to risk-off days.
  - Replayed through past crises it loses 3–10× VaR: SNB −$366k, COVID −$466k, Aug-2024 yen unwind −$155k.
  - VaR over the worst 500-day window in 12 years is $75.9k (vs $44.6k today).
  - There is no stress-loss or drawdown limit.
- **Hedges depend on correlations,** which change in a crisis (short CHF hedges in calm markets but loses in risk-off).
- **Component and marginal VaR are estimates.** VaR jumps from one replayed day to another, so for small hedges the
  real change can even have the other sign (GBPUSD +$0.1m: VaR +$56 vs −$165 predicted). Marginal ES is exact only
  for small changes. The pre-trade check rebuilds everything for real trade sizes.
- **The backtest** uses today's book over the past year, not the book's actual past P&L.
- **Data (Yahoo, unofficial):** today's price can't be cleaned (no next day yet); multi-day errors are only flagged;
  in older daily data the weekend move sits in Friday's return (each move is still counted once, so VaR is
  unaffected); CNH and HKD are excluded.
- **Seasonality** has about 10 observations per month: a 70% hit rate happens by luck 17% of the time.
- **Session / weekday P&L** covers only the hourly history Yahoo provides; P&L of older trades before it is shown
  separately.
- **Platform:** single user, no logins; on Streamlit Cloud the book file is shared by all viewers and resets when
  the app restarts; a CSV open in Excel can block saving (the dashboard flags it).

---

## 3. What I would improve with more time
1. **Stress testing:** day-by-day replays of 2008, SNB 2015, Brexit 2016, COVID 2020, the 2022 gilt crisis and the
   2024 yen unwind; stressed VaR; crisis correlations; reverse stress tests; hypothetical shocks (intervention, peg
   break, EM sudden stop) with a **stress-loss limit**.
2. **Risk governance:** drawdown / stop-loss limits; concentration, gross and factor limits (risk-on / off, carry,
   USD); EWMA or filtered historical simulation; formal backtest tests (Kupiec, Christoffersen).
3. **Trading costs:** bid / ask and volume by pair, session and hour (EBS, Refinitiv, LMAX, CME); spread spikes at
   the Monday open, NY 5pm rollover, fixes, data releases and month-end; a cost model (half-spread + impact) in the
   pre-trade check and P&L net of costs; direct vs via-USD routing for crosses; a liquidity add-on to VaR.
4. **Cross-asset and flows:** rolling β to equities, rate spreads, commodities (oil–CAD / NOK, copper–AUD,
   gold–ZAR) and VIX; equity and bond flows (Korea / Taiwan foreign flows, TIC, Japan MoF); CFTC positioning;
   rate-differential fair value.
5. **P&L and data:** carry and forward points, NDFs, realised vs unrealised P&L; licensed data with true NY closes;
   a database for the blotter; logins and multiple books.

---

# Appendix

### A. Trading universe
26 currencies vs USD (`universe.py`): **G10** EUR, GBP, CHF, JPY, AUD, NZD, CAD, NOK, SEK · **Asia EM** KRW, TWD,
SGD, THB, MYR, IDR, INR, PHP · **CEEMEA** PLN, HUF, CZK, ZAR, TRY · **LatAm** MXN, BRL, CLP, COP.
The ticket offers 26 pairs (EURUSD, GBPUSD, AUDUSD, NZDUSD, USDJPY, USDCHF, USDCAD, USDSEK, USDNOK, EURGBP, EURJPY,
EURCHF, GBPJPY, AUDJPY, AUDNZD, EURAUD, EURNOK, EURSEK, USDSGD, USDKRW, USDINR, USDMXN, USDZAR, USDBRL, USDPLN,
USDTRY); any pair of these currencies can be loaded from the CSV. Crosses are built from the USD legs (they match
Yahoo's quoted G10 crosses to ~1 bp).

### B. Code map
```
data/portfolio.csv ⇄ trades.py (read, validate, save) ◄── ticket / blotter     yfinance → market_data.py (re-date, clean)
positions.py (legs, exposure) → pnl.py (MTM) → risk.py (VaR, ES, splits) → engine.build_book() → Book
attribution.py · crosscheck.py · analytics/ (beta, richcheap, peers, sessions, seasonality, calendar_events)
views.py draws the tabs (Overview | Positions & risk | Analytics | Diagnostics); notes.py holds the 📘 notes; app.py wires it
```

| Change | File → function / constant |
|---|---|
| VaR / ES method, confidence, window | `risk.py` → `CONF`, `LOOKBACK`, `VAR_BAND`, `compute_risk` |
| Limits and flags | `engine.py` → `Settings`, `build_alerts` |
| P&L, exposure, break-even | `pnl.py` → `mtm_usd`, `trade_pnl`; `positions.py` → `exposure_by_currency`, `positions_table` |
| Book file and validation | `trades.py` → `read_book`, `save_book`, `validate_ticket`; `app.py` → `sync_book` |
| Attribution | `attribution.py` → `entry_legs`, `pnl_by_currency`, `session_attribution` |
| Data download and cleaning | `market_data.py` → `restamp_daily`, `clean_spikes`, `last_complete_date` |
| Universe and peer groups | `universe.py` → `CURRENCIES`, `PEER_GROUPS`, `TRADABLE_PAIRS` |
| Session windows / clock | `clock.py` → `SESSION_HOURS_UTC`, `trading_day_now` |

### C. Data handling
- **Load:** 26 USD pairs, 3 years daily plus 7 days hourly; cached 15 minutes; bundled fallback snapshot
  (`data/fallback_prices.csv`); prices older than 3 business days are flagged.
- **Cross check:** `gap = quoted cross ÷ cross built from USD legs − 1`, using only reliable references (typical gap
  ≤ 5 bps); a gap > 50 bps today, or an unexplained past gap > 200 bps, is flagged.
- Seasonality uses 12 years of daily data; sessions and attribution use Yahoo's hourly history.

### D. Run, test, logs
- **Local:** `streamlit run app.py` (`FXDASH_OFFLINE=1` uses the bundled snapshot). **Deploy:** GitHub →
  share.streamlit.io → main file `app.py`.
- **Tests:** `pytest -q` runs 74 offline tests (P&L, risk, attribution, analytics, data checks, book-file sync, app
  smoke test) on a temporary book file; `FXDASH_NETWORK_TESTS=1 pytest tests/test_network.py` runs the live
  date-fix check.
- **Logs:** `logs/app.log` (SGT); an error shows an ID to search in 🛠 Diagnostics.
