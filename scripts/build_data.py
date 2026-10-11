"""
One-off helper: refresh the bundled data files from Yahoo.

    python scripts/build_data.py            # fallback snapshots only
    python scripts/build_data.py --seed     # also rewrite data/sample_portfolio.csv at real historical closes

Outputs
  data/fallback_prices.csv   daily Q panel used when Yahoo is unavailable
  data/sample_portfolio.csv  sample trades (entry price = close on the trade date); 'Reset to sample' copies it
                             into data/portfolio.csv, the live book
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import market_data as M  # noqa: E402
import universe as U  # noqa: E402

# (trade_id, trade_date, pair, side, notional, purpose)
SEED = [
    ("T001", "2026-03-02", "EURUSD", "BUY", 5_000_000, "USD-quoted pair"),
    ("T002", "2026-04-15", "USDJPY", "BUY", 3_000_000, "USD-base pair"),
    ("T003", "2026-05-20", "GBPUSD", "SELL", 2_000_000, "short sign"),
    ("T004", "2026-06-10", "AUDUSD", "BUY", 4_000_000, "commodity bloc"),
    ("T005", "2026-07-01", "USDCAD", "SELL", 2_500_000, "USD-base short"),
    ("T006", "2026-08-12", "EURGBP", "SELL", 1_500_000, "cross pair"),
    ("T007", "2026-09-03", "USDCHF", "BUY", 2_000_000, "safe haven"),
    ("T008", "2026-09-22", "EURUSD", "SELL", 2_000_000, "partial close"),
    ("T009", None, "NZDUSD", "BUY", 3_000_000, "latest trading day"),
]


def main(write_seed: bool) -> None:
    md = M._load_live()
    if md.source != "live":
        raise SystemExit("Live download failed - not overwriting snapshots")
    M.DATA_DIR.mkdir(exist_ok=True)
    md.q.to_csv(M.FALLBACK_DAILY, float_format="%.10g")
    print(f"Saved {M.FALLBACK_DAILY.name}: {md.q.shape}, last date {md.last_date.date()}")

    if write_seed:
        rows = []
        for tid, date, pair, side, notional, _purpose in SEED:
            d = pd.Timestamp(date) if date else md.last_date
            price = md.pair_price(pair, md.q_on(d))
            dp = 3 if U.price_decimals(pair) == 2 else 5
            rows.append({"trade_id": tid, "timestamp": f"{d.date()} 12:00:00 UTC", "trade_date": d.date(),
                         "pair": pair, "side": side, "notional": notional, "price": round(price, dp),
                         "status": "LIVE"})
        seed = pd.DataFrame(rows)
        seed.to_csv(ROOT / "data" / "sample_portfolio.csv", index=False)
        print(seed.to_string(index=False))


if __name__ == "__main__":
    main(write_seed="--seed" in sys.argv)
