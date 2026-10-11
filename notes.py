"""
'How it works' notes shown at the top of each tab / view (ui.method_note renders them).

Each note has a plain-English part (what it is, why a PM looks at it, how to read it) and the exact formulas,
each with the file and function that computes it, so the explanation and the code can be checked side by side.
All times are SGT (UTC+8). Q_c = USD value of one unit of currency c (Q_USD = 1).
"""
from __future__ import annotations

# Symbols used in several notes, explained once in plain words.
_GLOSSARY = """
**Reading the symbols**
- **Q_c**: the price of one unit of currency c in US dollars. Q_EUR = 1.12 means 1 euro = 1.12 dollars;
  Q_JPY = 0.0063 means 1 yen = 0.0063 dollars (USDJPY 158). Q going **up** = the currency got **stronger**.
- **ln**: the natural log. **Δln Q = ln(Q today / Q yesterday)** is the day's move: about the same as the % change
  for normal days (a 1.00% rise is a 0.995 'log' rise).
- **Why ln instead of plain %?** (1) Log moves **add up** over days and split exactly into parts (dollar + bloc +
  own); plain % moves compound and leave small leftovers. (2) They are **symmetric**: if USDJPY rises 1% in log
  terms, JPYUSD falls exactly 1%, so flipping a pair just flips the sign.
"""

_RISK_METHOD = """
**Why historical simulation only (no parametric VaR)?** A parametric VaR assumes daily P&L follows a bell curve.
FX has fat tails and jumps (SNB 2015, BoJ interventions), so a bell curve understates bad days, and its risk split
rests on correlations that change in a crisis. Historical simulation simply replays what really happened: no
distribution is assumed, and the same replayed days give VaR, ES and the split between positions.

**Why these settings (fixed): 1 day, 95%, last 500 completed days.** 500 days (~2 years) covers more than one
calm year but is still recent; at 95% there are **25 tail days**, enough for a stable VaR and ES (at 99% only 5
days would decide everything); 1 day matches how the book is marked and managed. Today's unfinished day is left out.
"""

NOTES: dict[str, dict[str, str]] = {
    "overview": {
        "title": "Overview: the headline numbers",
        "body": """
**Words used below:** a currency's **USD price** (written Q) is how many US dollars one unit buys: EUR 1.1206,
JPY 0.0063. A trade's **signed notional** is + the amount for a BUY and − for a SELL (in the first currency of
the pair). The **previous close** is the price snapshot at **08:00 SGT** that ends each trading day.

#### Daily P&L
What the book made or lost since this morning's 08:00 SGT close. Each trade is valued twice, at today's prices and
at the close, and the difference is its daily P&L. A trade done after the close counts all of its P&L.
```python
out["inception_pnl"] = mtm_usd(out, q_now)        # each trade's USD value at today's prices
prev_mtm = mtm_usd(out, q_prev)                   # the same trade valued at the previous close
new_today = pd.to_datetime(out["trade_date"]) > pd.Timestamp(prev_date)   # done after that close?
out["daily_pnl"] = np.where(new_today, out["inception_pnl"], out["inception_pnl"] - prev_mtm)
```
- `q_now` / `q_prev`: today's / the previous close's USD price of every currency. `prev_date`: that close's day.

#### Inception P&L (and how one trade is valued)
A trade is two amounts: buying N EURUSD at P = **+N EUR** and **−N × P USD**. Its value is each amount × its
currency's USD price, added up. Inception P&L = that value today, summed over all live trades.
```python
return live["signed"] * _q(q_row, live["base"]) - live["signed"] * price * _q(q_row, live["quote"])
```
- `live["signed"]`: signed notional N. `price`: the trade price P.
- `_q(q_row, live["base"])` / `_q(q_row, live["quote"])`: today's USD price of the first / second currency.
- *Example: long 1m EURUSD at 1.10, EUR now 1.12 → 1m × 1.12 − 1m × 1.10 × 1 = +20,000 USD.*

#### VaR 95%, 1 day
Replay each of the last **500 completed days**' real currency moves on **today's** positions: 500 possible
"tomorrows". VaR = the loss beaten only on the worst 5% of them (about 1 day in 20). With 500 days that is
between the 25th and 26th worst day (the percentile interpolates between them).
```python
var_hs = hs_var(p, conf)
return float(-np.percentile(arr, (1 - conf) * 100))
```
- `p` (`arr`): the book's P&L on each of the 500 replayed days (how it is built: see *Who uses the risk budget*).
- `conf`: 0.95, so this is the 5th percentile of those 500 results, turned into a positive loss.

#### Expected shortfall 95%
VaR says where the bad days start; ES says how bad they are **on average**: the mean loss of the 25 worst days.
```python
return max(1, math.ceil(round((1 - conf) * n, 9)))   # k: n = 500 replayed days → k = 25 worst days
tail = p.nsmallest(k).index                       # the 25 days with the biggest book losses
es = float(-p.loc[tail].mean())                   # their average, as a positive loss
```

#### Gross exposure
The total size of all currency bets, ignoring whether long or short.
```python
gross=float(exposure_nonusd.abs().sum())          # USD value held in each non-USD currency, signs removed
```
""" + _RISK_METHOD,
        "formulas": "",
    },
    "exposure": {
        "title": "What am I exposed to?",
        "body": """
#### Net exposure by currency (left chart)
**What it shows:** for each currency, the USD value of everything you own (+) or owe (−) across all your trades.
*"AUD +$2.80m" = you own AUD worth USD 2.80m today; if AUD rises 1% vs USD you gain about $28k.*

**How it is built, step by step**
1. Every trade becomes two amounts: buying N EURUSD at P adds **+N EUR** and **−N × P USD**. A sell is the same with N negative.
2. All amounts of the same currency are added up, whenever the trades were done: buys add, sells subtract.
   *Example: buy 5m EURUSD (+5m EUR), sell 2m EURUSD (−2m EUR), buy 1m EURUSD (+1m EUR), sell 1.5m EURGBP
   (−1.5m EUR) → you hold EUR 2.5m.*
3. That amount is valued at **today's** USD price: EUR 2.5m × 1.1206 = **$2.80m**.
```python
t["signed"] = np.where(t["side"] == "BUY", 1.0, -1.0) * t["notional"].astype(float)
"amount": live["signed"]                                   # first currency of the pair: +N
"amount": -live["signed"] * live["price"].astype(float)    # second currency: −N × trade price
legs["usd"] = legs["amount"] * legs["ccy"].map(_q_full(q_row))
return legs.groupby("ccy")["usd"].sum().sort_values()
```
- `t["side"]`, `t["notional"]`: BUY/SELL and size from the blotter. `live["price"]`: the trade price.
- `legs["ccy"].map(_q_full(q_row))`: today's USD price of each leg's currency (USD itself = 1).

**Why today's price is the right one:** tomorrow's P&L from a currency is exactly *amount held × change in its
price*. The prices you traded at only decide how much of the other currency you owe; the gains and losses they
created are already in your P&L and cannot move again. **USD (base)** is the dollar side of your trades: your P&L
is counted in USD, so it carries no FX risk (grey bar).

#### Who uses the risk budget: contribution to ES (right chart)
**The question it answers:** *"When this book has a really bad day, which positions cause the loss?"* Each bar is
a position's **average loss on the book's 25 worst days** out of the last 500. The bars **add up exactly to ES**.

**How it is calculated, in 3 steps**
1. **Replay.** For each of the last 500 completed days, compute what each position would make or lose if that
   day's currency moves happened again today: its USD exposure in each currency × that currency's move that day.
2. **Find the bad days.** Add the positions up for each day to get the book's result, and pick the 25 worst days.
3. **Average each position over those 25 days.** That average loss is the position's bar.
```python
return pd.DataFrame(r.values @ exposure[ccys].values.T, index=r.index, columns=exposure.index)   # X[day, position]
p_full = x_full.sum(axis=1)               # the whole book's P&L on each day
p = p_full.iloc[-lookback:]               # keep the last 500 days
tail = p.nsmallest(k).index                # the 25 worst book days (k = 25)
component = -x.loc[tail].mean()            # each position's average P&L on those days, as a loss
```
- `r`: each currency's daily % move vs USD (`Q today / Q yesterday − 1`), completed days only.
- `exposure`: today's USD exposure of each position (pair) in each currency.
- Why the bars add up to ES: ES is the average of the book's P&L on those 25 days, and the book's P&L is the sum of
  the positions' P&L. The average of a sum is the sum of the averages.

**Worked example (current book: 10 trades, prices to the 9 Oct 2026 close; ES = $69.6k)**
- **AUDUSD** holds $2.80m of AUD. On the 25 worst days AUD fell 1.08% on average → about **$30.1k** of loss =
  **43%** of ES. It is the biggest user of the risk budget.
- **EURUSD** is the largest position ($4.48m of EUR), but EUR fell only 0.36% on those days → **$16.2k** (23%).
  Big is not the same as risky.
- **GBPUSD** is short $2.65m of GBP. GBP fell 0.39% on those days, so this position **gained** ~$10.3k →
  **negative bar: a hedge** (−15%). It cuts ES by $10.3k.
- **EURGBP** is short EUR and long GBP; both fell by similar amounts on those days, so it nearly cancels: $0.5k.

**Why is CAD the 3rd-largest exposure but USDCAD only 4th in risk, behind the smaller NZD?** A position's risk ≈
**its size × how much its currency moves on the book's bad days**, not its size alone. CAD is a calm currency
(typical daily move ~0.34% vs ~0.61% for NZD and 0.60% for AUD) and fell only ~0.50% on the 25 worst days, vs
~1.00% for NZD. So CAD $2.49m × 0.50% ≈ $12.5k, while the smaller NZD $1.68m × 1.00% ≈ $16.9k.

**How to use it:** to cut ES fastest, trim the **longest bar**. Adding to a **negative bar** lowers ES. Use the
pre-trade check to see the exact effect of a trade before booking it.

**Keep in mind:** the bars come from the last two years' 25 worst days. In a different kind of crisis different
positions can lead, and a hedge can stop hedging (for example, short CHF hedged in calm markets but loses when
CHF rallies in a risk-off move).
""",
        "formulas": "",
    },
    "pnl": {
        "title": "Where am I making or losing money?",
        "body": """
#### P&L by currency (exact; adds up to the headline Daily and Inception P&L)
Each trade is split into its two currency amounts; an amount makes money only when **its** currency moves against
the USD. USD amounts never make FX P&L. No fees or slippage are modelled, so there is nothing left over.

**Entry price of each currency.** A trade only fixes the pair price P, so each currency's USD entry price is
implied by it:
- AUDUSD at 0.69942 → AUD bought at **$0.69942**. USDJPY at 158.809 → JPY sold at **1 ÷ 158.809 = $0.006297**.
- A cross, EURGBP at 0.85421: GBP's entry = its USD price at that day's close; EUR's entry = 0.85421 × that.
```python
qb, qq = price, 1.0                    # xxxUSD: first currency at the trade price, USD = 1
qb, qq = 1.0, 1.0 / price              # USDxxx: second currency at 1 / trade price
qq = float(q0[t.quote])                # cross: second currency at its USD price on the trade-date close ...
qb = price * qq                        # ... first currency at trade price × that
start = q_entry if new else q_prev[c]  # daily: from the previous close (a trade done since then: from its entry)
rows.append((c, a * (q_now[c] - start), a * (q_now[c] - q_entry)))   # (currency, daily, inception)
```
- `price`: the trade price P. `q0`: USD prices at the trade date's close. `a`: the amount of currency `c` in the
  trade (+N for the first currency, −N × P for the second). `q_now` / `q_prev`: USD prices now / at the previous close.
- Because Q_first,entry ÷ Q_second,entry = P, the two amounts are worth exactly 0 at entry, so the currency rows add
  up to each trade's P&L.
- **Consistency with Positions:** a pair against USD puts all of its P&L in its currency (the NZD row = the NZDUSD
  position; JPY = USDJPY). Only crosses are split: EURGBP's P&L sits partly in EUR and partly in GBP.

#### P&L by session (Asia 08:00–16:00, Europe 16:00–21:00, US 21:00–08:00 SGT) and by weekday
For every hour: the amounts you held × how much each currency moved in that hour, grouped into sessions and
weekdays. A trade enters in the hour it was done (its booked-at time; a back-dated trade, or one with no time, at
its trade date's close). In that hour it makes the gap between its entry price and that hour's price; after that
it makes each hour's move.
```python
pos = int(idx.searchsorted(when, side="right")) - 1     # the hourly bar the trade was done in
entry_pnl[pos, j] += a * (h.iloc[pos, j] - q_entry)     # entry price → that hour's price
start_amt[pos + 1, j] += a                              # held from the next hour on
bar_pnl = pd.DataFrame(hold * moves + entry_pnl, index=idx, columns=ccys)
```
- `when`: when the trade was done (UTC). `h`: each currency's hourly USD price. `hold`: amount held in each hour
  (running total of `start_amt`). `moves`: each currency's change in USD price over each hour.

Sessions + any P&L before the hourly history = **inception P&L exactly** (shown under the charts). Monday includes
the weekend gap (the market reopens Monday ~05:00 SGT; those first hours are counted in Monday's Asia session).
""",
        "formulas": "",
    },
    "markets": {
        "title": "Markets: FX pairs of interest",
        "body": """
The price of each chosen pair over the chosen history, and its change over the last day (**CoD**), week (**CoW**)
and month (**CoM**), measured from the latest price (live while the trading day is still running) back 1, 5 and
21 trading days. The bar charts rank the pairs by weekly and monthly change. + = the first currency got
stronger (EURUSD up = euro up).
```python
chg = {p: {k: (prices[p].dropna().iloc[-1] / prices[p].dropna().iloc[-1 - d] - 1) * 100
           for k, d in (("CoD", 1), ("CoW", 5), ("CoM", 21))} for p in picks}
```
- `prices[p]`: the pair's daily price, built from the USD prices (EURGBP = EURUSD ÷ GBPUSD).
- `iloc[-1]`: today; `iloc[-1 - d]`: d trading days ago (1, 5 or 21).
""",
        "formulas": "",
    },
    "positions": {
        "title": "Positions: what do I hold, and at what price?",
        "body": """
#### The positions table (one row per pair; all live trades in a pair are netted)
- **Net notional**: buys minus sells, in the first currency. *Buy 5m EURUSD, sell 2m, buy 1m → net long 4m.*
- **Avg entry (break-even)**: the price at which the pair's total P&L would be zero:
  **S\\* = (Σ N × P) ÷ (Σ N)**. *Example (EURUSD: buy 5m at 1.1697, sell 2m at 1.1448, buy 1m at 1.1206):
  (5m × 1.1697 − 2m × 1.1448 + 1m × 1.1206) ÷ (5m − 2m + 1m) = 1.1699.* The loss already locked in on the 2m sold
  must be won back by the 4m still held. A fully closed pair has none (–).
- **Price**: today's price of the pair. **USD size**: net notional × today's USD price of the first currency.
```python
g = live.assign(np_=live["signed"] * live["price"].astype(float)).groupby("pair")
"net_notional": g["signed"].sum(),        # Σ N
"sum_np": g["np_"].sum(),                 # Σ N × P
out["avg_entry"] = np.where(out["net_notional"] != 0, out["sum_np"] / out["net_notional"].replace(0, np.nan), np.nan)
out["price_now"] = out["base"].map(q) / out["quote"].map(q)
out["size_usd"] = out["net_notional"] * out["base"].map(q)
```
- `live["signed"]`: signed notional N of each trade. `live["price"]`: its trade price P.
- `out["base"].map(q)` / `out["quote"].map(q)`: today's USD price of the first / second currency.
- **Daily / inception P&L** per pair = the sum over its trades (see Overview). **Position VaR, component VaR,
  marginal VaR, contribution to ES, % of risk, marginal ES**: see the *Risk* note above (they come from the same
  replayed days as the VaR and ES tiles).

#### The blotter (every transaction)
Every trade ever booked: id, booked-at time (SGT), trade date, pair, BUY/SELL, notional, price and status. It is
the single source of truth: positions are rebuilt from it on every click, and a cancelled trade stays, greyed out.

**The blotter is a file, synced both ways: `data/portfolio.csv`.** Booking, cancelling, loading or resetting here
writes the file at once; editing the file yourself (e.g. in Excel) shows up here within a few seconds. A file with
mistakes is not loaded (the last good book stays and the error names the row). A row without a trade_id gets the
next free id. Reset to sample / Load file keep the previous file as `portfolio.backup.csv`.
""",
        "formulas": "",
    },
    "risk": {
        "title": "Risk: how much could I lose, and why?",
        "body": """
**One method throughout: historical simulation, 1 day, 95%, last 500 completed days.** Everything below comes from
replaying those 500 days on today's positions (the replay is explained step by step in Overview → *Who uses the
risk budget*).
```python
returns = md.simple_returns(list(pair_exp.columns)).loc[:md.last_complete_date]
return pd.DataFrame(r.values @ exposure[ccys].values.T, index=r.index, columns=exposure.index)
```
- `returns` / `r`: each currency's daily % move vs USD, completed days only (today's unfinished day is left out).
- `exposure`: today's USD exposure of each position in each currency. The result **X[day, position]** = what each
  position would make or lose if that day happened again; **p** = their sum = the book's P&L that day.

#### Historical VaR 95% and Expected shortfall 95%
VaR = the loss beaten on only the worst 5% of the 500 days; ES = the average of the 25 worst days.
*Current book: VaR $44.6k (89% of the $50k limit), ES $69.6k.*
```python
var_hs = hs_var(p, conf)                          # -percentile(p, 5%)
tail = p.nsmallest(k).index                       # k = 25 worst days of 500
es = float(-p.loc[tail].mean())
```

#### Diversification benefit
Each position's own VaR, added up, minus the book's VaR: positions rarely all lose on the same day.
*Current book: $147.8k − $44.6k = $103.2k.*
```python
position_var = x.apply(lambda col: hs_var(col, conf))     # VaR of each position on its own
diversification=float(position_var.sum() - var_hs)
```

#### VaR backtest
For each of the last 250 days: the VaR computed from the 500 days **before** it, vs what today's book would have
lost that day. A **breach** = the loss was bigger than VaR. About 12–13 are expected at 95%; many more means VaR
is too low. *Current book: 4.*
```python
var_i = hs_var(vals[i - lookback:i], conf)                 # VaR from the 500 days before day i
rows.append((full_pnl.index[i], vals[i], var_i, vals[i] < -var_i))   # breach if that day lost more
```

#### Risk columns in the Positions table (below)
All six come from the same replay table **X[day, position]**; *USD size* = |net notional × today's USD price of
the first currency| (AUDUSD $2.80m).
- **Position VaR**: the pair's own VaR, as if it were the only position: the loss beaten on only 5% of the 500 days
  of its own replayed P&L. *AUDUSD $23.8k.*
```python
position_var = x.apply(lambda col: hs_var(col, conf))     # each column of X on its own: -percentile(5%)
```
- **Component VaR**: each pair's share of the book's VaR; the column adds up to VaR. VaR is set by the 25th/26th
  worst day, and one day's split is noisy, so it uses the **11 days ranked around it (20th–30th worst)**: each pair's
  average loss on those days, scaled so the column adds up to VaR exactly. *AUDUSD $17.2k of VaR $44.6k; GBPUSD
  −$4.4k (hedge).*
- **Marginal VaR / $1m** = component VaR ÷ |USD size| × 1m: roughly how much VaR moves per $1m more of the position.
  *AUDUSD: $17.2k ÷ $2.80m × 1m ≈ $6.1k (rebuilt with $0.1m more: VaR +$650 vs +$615 predicted).* It is a guide
  only: VaR jumps from one replayed day to another, so for small hedges the real change can even have the other sign
  (GBPUSD +$0.1m: VaR +$56 vs −$165 predicted). That is why the ES columns are the main risk split.
```python
mid = int(math.floor((1 - conf) * (len(p) - 1)))           # rank of the VaR day (24 = 25th worst of 500)
band = p.index[order[max(0, mid - VAR_BAND): mid + VAR_BAND + 1]]   # the 11 days ranked 20th-30th worst
raw = -x.loc[band].mean()
component_var = raw * (var_hs / raw.sum()) if abs(raw.sum()) > 1e-9 else raw * 0
marginal_var = component_var / size.replace(0, np.nan) * 1e6
```
- **Contribution to ES**: its average loss on the book's 25 worst days; the column adds up to ES. Negative = hedge.
- **% of risk** = contribution ÷ ES. *AUDUSD: $30.1k ÷ $69.6k = 43%.*
- **Marginal ES / $1m** = how much ES would rise if this position were $1m bigger (small changes; for a real
  trade size use the pre-trade check). *AUDUSD: $30.1k ÷ $2.80m × 1m ≈ $10.8k.*
```python
component = -x.loc[tail].mean()                            # contribution to ES
pct = component / es if es else component * 0              # % of risk
marginal = component / size.replace(0, np.nan) * 1e6       # size = |net notional × USD price of the first currency|
```

**Why can Position VaR be positive while Contribution, % of risk and Marginal ES are negative (a hedge)?**
They answer different questions.
- **Position VaR** asks: *"how much can this position lose on its own bad days?"* Every position that can lose
  has a positive VaR. *GBPUSD (short $2.65m of GBP) loses when GBP rallies: about $19.5k on its own worst-5% days.*
- **Contribution to ES** asks: *"what does this position do on the **book's** bad days?"* The book is long AUD,
  NZD, CAD and EUR, so its worst days are days when those currencies fall, and GBP usually falls with them
  (−0.39% on average). Short GBP **made** money on 19 of those 25 days, about +$10.3k on average, so its
  contribution is −$10.3k. GBPUSD's bad days (GBP up) are mostly the book's good days.
- **% of risk** = −$10.3k ÷ $69.6k = **−15%**: it offsets 15% of the book's tail loss.
- **Marginal ES** = −$10.3k ÷ $2.65m × 1m = **−$3.9k**: a bit **more** of this short would **lower** ES. Checked
  by rebuilding the book: selling $0.1m more GBPUSD cuts ES by $387, as predicted. For $1m it cuts ES by $3.5k
  (not the full $3.9k): bigger trades start to change which days are the worst. Use the pre-trade check for real sizes.
- **Keep in mind:** it is a hedge only for this book's past bad days. In a crisis where GBP rallies while AUD and
  NZD fall, it would add to the loss. Its Position VaR is the reminder that it can still lose on its own.

#### Correlations (model checks)
How the held currencies' daily moves went together over the same 500 days (+1 = always together, −1 = opposite).
```python
corr = md.simple_returns(ccys).loc[:md.last_complete_date].iloc[-r.lookback:].corr()
```
""" + _RISK_METHOD,
        "formulas": "",
    },
    "pre_trade": {
        "title": "Pre-trade check: what this trade would do",
        "body": """
#### Before / after table
The proposed trade is added to a copy of the blotter and **the whole book is rebuilt** with it, so every number is
recalculated exactly, not estimated. Nothing is booked until you confirm.
```python
trial = T.with_trade(ss["trades"], T.make_trade(ss["trades"], p["pair"], p["side"], p["notional"], p["price"], p["trade_date"]))
after = E.build_book(trial, md, betas, settings)           # the full book again, with the new trade
df["Change"] = df["After"] - df["Before"]
```
- Rows: **VaR** and **ES** (as in Positions & risk), **gross exposure**, **DXY-equivalent** (below), and the USD
  exposure of the two currencies traded. *Buying 1m EURUSD adds about $1.12m of EUR.*

#### VaR after trade (green / orange / red)
```python
used = after.risk.var_hs / settings.var_limit              # green below 80%, orange 80–100%, red above the limit
```

#### Market context line (for each currency bought or sold)
*Example: "Buying EUR: −2.8σ vs 1Y, −1.6σ vs peers, 1M move −2.7σ, USD β 0.93" reads as: EUR is unusually cheap
vs its last year, weak vs its peers after removing the dollar, has had an unusually large 1-month fall, and moves
almost one-for-one with the dollar index.* In σ (standard deviations): 0 = normal, ±1 = notable, ±2 = rare.
In currency terms: + = the currency is strong / expensive.

- **vs 1Y (level z)**: today's price vs its average of the previous 252 trading days, in standard deviations.
```python
prior = log_q.shift(1).rolling(window, min_periods=int(window * 0.8))   # the previous 252 days, today excluded
z = (log_q - prior.mean()) / prior.std()
```
  `log_q` = ln of the currency's USD price; `window` = 252.

- **vs peers (peer z, 3 months)**: the currency's own move after removing the dollar and its bloc (e.g. AUD vs NZD,
  CAD, NOK), added up over time and compared with its previous 63 days.
```python
cum = own.fillna(0.0).cumsum()                     # running total of the currency's own daily moves
prior = cum.shift(1).rolling(window, min_periods=int(window * 0.8))      # previous 63 days
return ((cum - prior.mean()) / prior.std()).iloc[-1]
```
  `own` = daily move − dollar part (β × DXY move) − bloc part (median of the peers' moves beyond the dollar).

- **1M move (move z)**: the last 21 trading days' move divided by the move a normal month would have, judged by
  the currency's volatility in the 63 days before.
```python
move = log_q.iloc[-1] - log_q.iloc[-1 - h]         # h = 21 trading days
vol = lr.iloc[-(h + vol_window):-h].std()          # typical daily move in the 63 days before that month
return move / (vol * np.sqrt(h))
```

- **USD β (63 days)**: for each 1% the dollar index (DXY) moves, how much USD moves against this currency.
```python
cov = usd_ret[c].rolling(window, min_periods=min_periods).cov(dxy_ret)
betas[c] = cov / var_d
```
  `usd_ret[c]` = daily move of USD vs the currency; `dxy_ret` = daily move of the DXY (rebuilt from the same
  prices); `var_d` = how much the DXY moves; `window` = 63 days.

- **For a cross** (e.g. EURGBP) a further line gives the cross's own level z (1Y, 3M) and 1M move z.

#### DXY-equivalent
The position in the dollar index that would move like your whole book: each currency's exposure × its β, added up
with the sign flipped. Negative = like being short the dollar; DXY +1% → book P&L ≈ DXY-equivalent × 1%.
```python
b = beta.reindex(exposure_nonusd.index).fillna(1.0)        # each held currency's USD β (1 if unknown)
return float(-(exposure_nonusd * b).sum())                 # −Σ exposure × β
```
*Buying 1m EURUSD adds about $1.12m of EUR × β 0.93 ≈ $1.05m more short-dollar: DXY-equivalent −$1.59m → −$2.63m.*

#### Fat-finger check
A price more than 5% away from the market (or from that day's close for a back-dated trade) needs confirming.
```python
if market_price and price and price > 0 and abs(price / market_price - 1) > PRICE_WARN:   # PRICE_WARN = 0.05
```
""",
        "formulas": "",
    },
    "rich_cheap": {
        "title": "Rich / cheap: level and move",
        "body": """
- **Level z**: is the price high or low **compared with its own recent average**? Measured over 1M, 3M, 6M and 1Y.
  +2 = about as expensive as it gets vs that period (roughly 1 day in 40 if prices were normal).
  1W is not shown for levels: an average of the last few days is almost today's price.
- **Move z**: was the move over 1W … 1Y **unusually big** for this currency? A 1% move is big for SGD, normal for ZAR,
  so moves are divided by each currency's own normal volatility.
- **% change**: the plain (simple) % move, currency vs USD (+ = currency stronger). Not a log move.
- Green = positive (strong / expensive), red = negative (weak / cheap).
""",
        "formulas": """\
Level z (N)   = (ln Q_t − mean(ln Q, prior N days)) / std(ln Q, prior N days)   N = 21, 63, 126, 252   analytics/richcheap.py: level_z
Move z (h)    = (ln Q_t − ln Q_t−h) / (σ_daily × √h),  σ_daily = std of the 63 daily log moves before   analytics/richcheap.py: stretch_z
% change (h)  = (Q_t / Q_t−h − 1) × 100     h = 5, 21, 63, 126, 252 trading days      analytics/richcheap.py: rich_cheap_table""",
    },
    "seasonality": {
        "title": "Seasonality: what does this pair usually do now?",
        "body": """
- Uses the number of **completed years** you type (default 10, up to 11) of daily closes. **All moves on this page
  are plain (simple) % changes**, not log moves. Every year is lined up by **day of the year** (in leap years, dates after February sit one day later) and
  starts from the **same kind of close**, so years can be compared like for like. Weekends and holidays carry
  the last close forward.
- **Path from a date** (default today): for each past year, the % move from the close on that date over the next
  N days (type any start date and number of days). Shows what *usually happens next* from this point in the calendar.
- **Month profile**: each year's move through this month, from the previous month-end close (day 0).
- **Year profile**: each year's move since the previous year-end close, with the 30th–70th percentile band.
- Lines: **grey = each earlier year**, **dotted blue / orange = the two latest completed years** (worked out from
  today's date, never fixed), **black = average**, **dashed = median**, **red = this year**, **green dotted = today**.
- **Monthly bars**: average and median monthly move per calendar month, with this year's months in red; the
  x-axis shows each month's **hit rate** (share of years it was up).
- **Heatmap**: every month of every year (1 dp, %), plus mean, median and hit rate. **Each row is coloured on its
  own scale** (green = up, red = down) so a quiet year's pattern is as visible as a volatile year's.
- **Demeaned** (optional, monthly bars and heatmap): each month minus the average of the previous 12 months, to
  remove the trend of that period using only past data.
- About 10 observations per month: treat patterns as hints, not rules.
""",
        "formulas": """\
path(y, t)     = (price on day t of year y / starting close of year y − 1) × 100
                 analytics/seasonality.py: forward_paths, month_paths, year_paths
average / median / 30th–70th pct: across the completed years at each calendar day     analytics/seasonality.py: summary
monthly r_m    = (close at month-end m / close at month-end m−1 − 1) × 100            analytics/seasonality.py: monthly_returns
demeaned r_m   = r_m − average(r_m−12 … r_m−1)                                       analytics/seasonality.py: demean_trailing
hit rate       = share of years with r > 0                                           analytics/seasonality.py: month_stats
charts         = views.py: _seasonality, _season_fig, _season_heatmap""",
    },
    "dxy_beta": {
        "title": "DXY beta: how much of the move is just the dollar?",
        "body": """
**The DXY**
- The **DXY** is the dollar index: the dollar against a basket of EUR (57.6%), JPY, GBP, CAD, SEK and CHF. DXY up =
  the dollar got stronger in general.
- It is **rebuilt here** from the same Yahoo FX prices with ICE's official formula. Why: Yahoo's own DXY series is
  stamped one day differently from its FX prices. *Example: the dollar jumps on Tuesday; Yahoo's FX prices show it
  on Tuesday but its DXY shows it on Wednesday. Compared day by day, the two would look unrelated and every beta
  would come out near zero.* Rebuilding the DXY from the same prices puts both on the same clock.

**Beta (β) and R²**
- **β_c (USD beta)**: for each 1% the DXY moves, how much does the dollar move against currency c?
  *Example: β_EUR = 0.93 → when DXY rises 1%, USD rises about 0.93% vs EUR. β_CAD = 0.65 → CAD only follows the
  dollar 65% as much.* β is measured from the last 63 days (3M) or 252 days (1Y) of daily moves.
- **R²**: how **reliably** the currency follows the dollar, from 0 (not at all) to 1 (perfectly).
- **Beta is not R²**. Beta = *how much* it moves with the dollar; R² = *how consistently*. *Example: JPY can have
  β = 1.4 (moves a lot when the dollar moves) but R² = 0.5 (on half the days other things, like rate news in Japan,
  dominate).* EUR has β ≈ 0.9 and R² ≈ 0.9: it moves almost one-for-one and very reliably, partly because EUR is
  57.6% of the DXY itself.

**The scatter chart**
- Each dot is a currency: across = its β (3M), up = how much USD moved against it over the last month (plain %).
- The **line** is 'what the dollar alone predicts': for any β on the x-axis, line = β × the DXY's 1-month move.
  *Example: DXY +2% over the month → a β = 1 currency should be at +2%, β = 0.5 at about +1%, β = 0 at 0%.* (The
  model works in log moves; converted to plain % the line bends very slightly.)
- **Above the line** = USD rose more against it than the dollar explains (the currency's own weakness);
  **below** = it held up better.

**Layered breakdown and DXY-equivalent**
- A currency's move = **dollar** part (β × DXY move) + **bloc** part (what its peers did beyond the dollar) + **own**
  part (its own story). Shown in **log %** because only log moves split into parts that add up exactly; the
  chart title also gives the plain % total.
- **DXY-equivalent** of your book (used in the pre-trade check and flags): the DXY position that would behave like
  the book. Negative = like being short USD.
""" + _GLOSSARY,
        "formulas": """\
DXY            = 50.14348112 × EURUSD^−0.576 × USDJPY^0.136 × GBPUSD^−0.119 × USDCAD^0.091 × USDSEK^0.042 × USDCHF^0.036
                 analytics/beta.py: synthetic_dxy
u_c            = −Δln Q_c        the dollar's daily move AGAINST currency c (+ = USD stronger vs c)
d              = Δln DXY         the dollar index's daily move
β_c (N days)   = cov(u_c, d) / var(d)        N = 63 (3M) or 252 (1Y)          analytics/beta.py: rolling_beta
R²_c           = corr(u_c, d)²                                               analytics/beta.py: rolling_beta
scatter line   = exp(β × ln(DXY_today / DXY_21 trading days ago)) − 1, in %  (dots: USD vs c, plain % over 21 days)
                 analytics/__init__.py: compute_market_analytics;  views.py: _dxy_beta
dollar part    = −β_c(yesterday) × d_t        (yesterday's β → no look-ahead)  analytics/beta.py: beta_analytics
bloc part      = median over the peers of (their move − their dollar part)    analytics/peers.py: bloc_and_own
own part       = Δln Q_c − dollar part − bloc part                            analytics/peers.py: bloc_and_own
DXY-equivalent = − Σ_c E_c × β_c             (DXY +1% → book P&L ≈ DXY-equivalent × 1%)   analytics/beta.py: dxy_equivalent""",
    },
    "peers": {
        "title": "Peers: is it the bloc or the currency itself?",
        "body": """
- Currencies move in groups: commodity currencies (AUD, NZD, CAD, NOK), Asia EM, CEEMEA, LatAm, G10 havens.
  Each currency is measured against **one** main group: NOK against the commodity bloc, so in the Scandies view
  it is labelled "NOK (vs G10 commodity)", and SEK's bloc is NOK alone (the Scandies have only two members).
- **Step 1, remove the dollar**: the part of today's move explained by the dollar index (β × DXY move).
- **Step 2, remove the bloc**: the median of what the *other* members did beyond the dollar (the currency itself is
  left out, so it cannot influence its own yardstick).
- **What is left is the currency's own move.** *Example: AUD −1.0% today; the dollar explains −0.6% and the
  commodity bloc another −0.3%, so AUD's own move is −0.1%: almost all of it was 'the market', not AUD.*
- **Bar chart and table**: in **log %** (so total = dollar + bloc + own exactly). Positive own move = beat its bloc.
- **Peer z**: the currency's cumulative own move vs its last 3 months (or 1 year), in standard deviations:
  + = unusually strong vs its peers.
""" + _GLOSSARY,
        "formulas": """\
move_c        = Δln Q_c                       (daily log move of currency c vs USD; + = c stronger)
dollar part   = −β_c(yesterday) × Δln DXY     (β_c = c's USD beta, see DXY beta)      analytics/beta.py: beta_analytics
residual_c    = move_c − dollar part
bloc_c        = median(residual of the other members of c's group)  (leave-one-out)  analytics/peers.py: bloc_and_own
own_c         = residual_c − bloc_c           →  move = dollar + bloc + own (exact)   analytics/peers.py: bloc_and_own
Peer z (N)    = (cum own_t − mean(cum own, prior N days)) / std(cum own, prior N days)   N = 63, 252   analytics/peers.py: peer_z
groups        = universe.py: PEER_GROUPS, PRIMARY_GROUP;  charts: views.py: _peers""",
    },
    "sessions": {
        "title": "Sessions: when in the day does the currency move?",
        "body": """
- The day is split into three **fixed SGT windows**: **Asia 08:00–16:00**, **Europe 16:00–21:00**,
  **US 21:00–08:00 (next day)**. A trading day runs 08:00 SGT to 08:00 SGT; the weekly reopen (Monday ~05:00 SGT)
  counts as Monday's Asia session.
- London and New York change clocks in summer, so their real opens move by one hour in SGT. The windows stay
  fixed so every day is measured the same way (in summer the first London hour counts as Asia).
- **Units**: session moves are **log %** so the three stack exactly into the day; the cumulative column is
  converted back to a plain % move.
- **Session move** = the log move from the start to the end of the window. **Cumulative** = the session moves added
  up over the period, shown as a % move. **Up sessions %** = how often that session was positive.
- **★ z-score**: today's session move vs the same session over the previous N days (type N below). |z| ≥ 1 is
  starred. Today is never part of its own yardstick.
- **Calendar**: holidays (HD) in either country, month-end (ME), mid-month (MM) and fiscal year start/end (FS/FE).
  Events on a weekend are shown on the next trading day. Holiday dates come from the open-source Python package
  **`holidays`** (PyPI; `holidays.country_holidays("JP")` etc.); ME / MM / FS / FE are computed from the calendar.
""",
        "formulas": """\
session move  = Σ hourly ln(P_h / P_h−1) over the window = ln(P_end / P_start)  (in %)   analytics/sessions.py: session_returns
day move      = Asia + Europe + US (exact)                                        analytics/sessions.py: session_returns
cumulative    = exp(Σ session moves) − 1;  hit rate = share > 0                  analytics/sessions.py: session_summary
session z     = (today − mean(prior N days)) / std(prior N days)   N typed (default 90)   analytics/sessions.py: session_zscores
windows       = clock.py: SESSION_HOURS_UTC, SESSION_SGT
calendar      = analytics/calendar_events.py: events_between, map_to_trading_days, upcoming;  charts: views.py: _sessions""",
    },
}
