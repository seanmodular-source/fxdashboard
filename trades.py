"""
The trade log (blotter) - the single source of truth. Positions are always rebuilt from it.

Storage: the book lives on disk in data/portfolio.csv and is kept in sync BOTH ways:
  * dashboard -> file: every booking, cancel, upload or reset is written to the CSV at once (save_book,
    atomic: write a temp file, then replace);
  * file -> dashboard: the app watches the file's signature (modified time + size) and reloads it when
    someone edits it (read_book). A file with errors is not loaded; the last valid book stays on screen.
data/sample_portfolio.csv is the untouched sample used by "Reset to sample". FXDASH_BOOK_FILE points the
book somewhere else (the tests use a temporary file so they never touch your book).
This module is pure pandas (no Streamlit) so it can be unit-tested; app.py / views.py own the wiring.
"""
from __future__ import annotations

import io
import math
import os
import shutil
from pathlib import Path

import pandas as pd

import clock
import universe as U
from logger import get_logger

log = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
SEED_FILE = DATA_DIR / "sample_portfolio.csv"     # the sample book ("Reset to sample"), never written by the app
COLUMNS = ["trade_id", "timestamp", "trade_date", "pair", "side", "notional", "price", "status"]
REQUIRED = ["trade_date", "pair", "side", "notional", "price"]
PRICE_WARN = 0.05  # warn if the trade price is more than 5% away from the market


def normalize(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate and clean a blotter (seed file or user upload). Returns (clean_df, errors)."""
    errors: list[str] = []
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return pd.DataFrame(columns=COLUMNS), [f"Missing column(s): {', '.join(missing)}"]

    if "trade_id" not in df.columns:
        df["trade_id"] = [f"T{i:03d}" for i in range(1, len(df) + 1)]
    else:  # a row typed into the CSV without an id gets the next free one; existing ids never change
        blank = df["trade_id"].isna() | (df["trade_id"].astype(str).str.strip().isin(["", "nan"]))
        if blank.any():
            nums = pd.to_numeric(df.loc[~blank, "trade_id"].astype(str).str.extract(r"(\d+)$")[0], errors="coerce")
            start = int(nums.max()) + 1 if nums.notna().any() else 1
            df["trade_id"] = df["trade_id"].astype(object)
            df.loc[blank, "trade_id"] = [f"T{start + i:03d}" for i in range(int(blank.sum()))]
    if "timestamp" not in df.columns:
        df["timestamp"] = ""
    if "status" not in df.columns:
        df["status"] = "LIVE"

    df["trade_id"] = df["trade_id"].astype(str).str.strip()
    df["pair"] = df["pair"].astype(str).str.upper().str.replace("/", "", regex=False).str.strip()
    df["side"] = df["side"].astype(str).str.upper().str.strip()
    df["status"] = df["status"].fillna("LIVE").astype(str).str.upper().str.strip()
    df["timestamp"] = df["timestamp"].fillna("").astype(str)
    df["notional"] = pd.to_numeric(df["notional"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()

    for i, row in df.iterrows():
        tag = f"Row {i + 1} ({row['trade_id']})"
        try:
            U.parse_pair(row["pair"])
        except ValueError as exc:
            errors.append(f"{tag}: {exc}")
        if row["side"] not in ("BUY", "SELL"):
            errors.append(f"{tag}: side must be BUY or SELL")
        if not (pd.notna(row["notional"]) and row["notional"] > 0 and math.isfinite(row["notional"])):
            errors.append(f"{tag}: notional must be a positive number")
        if not (pd.notna(row["price"]) and row["price"] > 0):
            errors.append(f"{tag}: price must be a positive number")
        if pd.isna(row["trade_date"]):
            errors.append(f"{tag}: trade_date is not a valid date")
        if row["status"] not in ("LIVE", "CANCELLED"):
            errors.append(f"{tag}: status must be LIVE or CANCELLED")
    if df["trade_id"].duplicated().any():
        errors.append("Duplicate trade_id values: " + ", ".join(df.loc[df["trade_id"].duplicated(), "trade_id"]))

    return df[COLUMNS].reset_index(drop=True), errors


def load_seed() -> pd.DataFrame:
    df, errors = normalize(pd.read_csv(SEED_FILE))
    if errors:
        log.error("Seed portfolio has errors: %s", errors)
        raise ValueError("Seed portfolio invalid: " + "; ".join(errors))
    log.info("Loaded seed portfolio: %d trades", len(df))
    return df


def book_path() -> Path:
    """The live book file: data/portfolio.csv, or FXDASH_BOOK_FILE if set."""
    return Path(os.environ.get("FXDASH_BOOK_FILE") or DATA_DIR / "portfolio.csv")


def book_signature() -> tuple[int, int] | None:
    """(modified time in ns, size) of the book file; changes whenever anyone saves it. None if missing."""
    try:
        s = book_path().stat()
    except FileNotFoundError:
        return None
    return (s.st_mtime_ns, s.st_size)


def read_book() -> tuple[pd.DataFrame, list[str]]:
    """Load and validate the book file (created from the sample if missing). Returns (clean_df, errors)."""
    path = book_path()
    if not path.exists():
        log.info("No book file at %s - creating it from the sample", path)
        save_book(load_seed())
    try:
        raw = pd.read_csv(path)
    except Exception as exc:
        return pd.DataFrame(columns=COLUMNS), [f"Could not read {path.name}: {exc}"]
    df, errors = normalize(raw)
    if errors:
        log.error("Book file %s has errors, not loaded: %s", path.name, errors[:5])
    else:
        log.info("Book loaded from %s: %d trades", path.name, len(df))
    return df, errors


def save_book(df: pd.DataFrame, backup: bool = False) -> tuple[int, int] | None:
    """Write the book to its file atomically (temp file, then replace). ``backup`` first copies the current file
    to <name>.backup.csv (used before 'Reset to sample' and 'Load file', which replace the whole book).
    Returns the new file signature. Raises OSError if the file cannot be written (e.g. open in Excel)."""
    path = book_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup and path.exists():
        shutil.copyfile(path, path.with_name(path.stem + ".backup.csv"))
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(to_csv_bytes(df))
    os.replace(tmp, path)
    log.info("Book saved to %s: %d trades", path.name, len(df))
    return book_signature()


def read_upload(file_bytes: bytes) -> tuple[pd.DataFrame, list[str]]:
    try:
        raw = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        return pd.DataFrame(columns=COLUMNS), [f"Could not read CSV: {exc}"]
    df, errors = normalize(raw)
    log.info("Blotter upload: %d rows, %d errors", len(df), len(errors))
    if errors:
        log.warning("Upload rejected: %s", errors[:5])
    return df, errors


def next_trade_id(df: pd.DataFrame) -> str:
    nums = pd.to_numeric(df["trade_id"].str.extract(r"(\d+)$")[0], errors="coerce")
    n = int(nums.max()) + 1 if nums.notna().any() else 1
    return f"T{n:03d}"


def validate_ticket(pair: str, side: str, notional: float, price: float, trade_date,
                    market_price: float | None, today: pd.Timestamp, has_prices: bool) -> tuple[list[str], list[str]]:
    """Ticket checks. Errors block the trade; warnings need the user's confirmation."""
    errors, warnings = [], []
    try:
        U.parse_pair(pair)
    except ValueError as exc:
        errors.append(str(exc))
    if not has_prices:
        errors.append(f"No market data for {pair}; cannot value or risk it")
    if side not in ("BUY", "SELL"):
        errors.append("Side must be BUY or SELL")
    if not notional or notional <= 0:
        errors.append("Notional must be greater than zero")
    if not price or price <= 0:
        errors.append("Price must be greater than zero")
    td = pd.Timestamp(trade_date).normalize()
    if td > today + pd.Timedelta(days=1):
        errors.append("Trade date cannot be in the future")
    if market_price and price and price > 0 and abs(price / market_price - 1) > PRICE_WARN:
        warnings.append(f"Price {price:,.5g} is {abs(price / market_price - 1):.1%} away from the market "
                        f"({market_price:,.5g}) - possible fat finger")
    return errors, warnings


def make_trade(df: pd.DataFrame, pair: str, side: str, notional: float, price: float, trade_date) -> dict:
    return {
        "trade_id": next_trade_id(df),
        "timestamp": clock.now_sgt().strftime("%Y-%m-%d %H:%M:%S SGT"),
        "trade_date": pd.Timestamp(trade_date).normalize(),
        "pair": pair.upper(), "side": side, "notional": float(notional), "price": float(price), "status": "LIVE",
    }


def with_trade(df: pd.DataFrame, trade: dict) -> pd.DataFrame:
    """Blotter plus one extra trade (used for the pre-trade preview - nothing is booked)."""
    return pd.concat([df, pd.DataFrame([trade])[COLUMNS]], ignore_index=True)


def add_trade(df: pd.DataFrame, trade: dict) -> pd.DataFrame:
    out = with_trade(df, trade)
    log.info("Trade booked: %s %s %s %s @ %s (date %s)", trade["trade_id"], trade["side"], trade["pair"],
             f"{trade['notional']:,.0f}", trade["price"], pd.Timestamp(trade["trade_date"]).date())
    return out


def cancel_trade(df: pd.DataFrame, trade_id: str) -> pd.DataFrame:
    out = df.copy()
    mask = out["trade_id"] == trade_id
    if not mask.any():
        log.warning("Cancel requested for unknown trade %s", trade_id)
        return out
    out.loc[mask, "status"] = "CANCELLED"
    log.info("Trade cancelled: %s", trade_id)
    return out


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.strftime("%Y-%m-%d")
    return out[COLUMNS].to_csv(index=False).encode("utf-8")
