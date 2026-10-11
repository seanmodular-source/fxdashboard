"""
Static configuration: which currencies exist, how to fetch them, and how they are grouped.

Convention used everywhere in the engine:
    Q[c] = USD value of 1 unit of currency c   (so + return = currency stronger vs USD)
    Q["USD"] = 1 by definition.

Yahoo quotes some pairs as CCYUSD (EURUSD=X -> already Q) and others as USDCCY
(USDJPY=X -> Q = 1 / price). ``invert`` records which.
"""
from __future__ import annotations

# code: (yahoo ticker, invert to get USD-per-unit, display name)
CURRENCIES: dict[str, tuple[str, bool, str]] = {
    # G10
    "EUR": ("EURUSD=X", False, "Euro"),
    "GBP": ("GBPUSD=X", False, "British pound"),
    "AUD": ("AUDUSD=X", False, "Australian dollar"),
    "NZD": ("NZDUSD=X", False, "New Zealand dollar"),
    "JPY": ("USDJPY=X", True, "Japanese yen"),
    "CHF": ("USDCHF=X", True, "Swiss franc"),
    "CAD": ("USDCAD=X", True, "Canadian dollar"),
    "SEK": ("USDSEK=X", True, "Swedish krona"),
    "NOK": ("USDNOK=X", True, "Norwegian krone"),
    # Asia EM (CNH excluded: Yahoo history is unavailable / stale)
    "KRW": ("USDKRW=X", True, "Korean won"),
    "TWD": ("USDTWD=X", True, "Taiwan dollar"),
    "SGD": ("USDSGD=X", True, "Singapore dollar"),
    "THB": ("USDTHB=X", True, "Thai baht"),
    "MYR": ("USDMYR=X", True, "Malaysian ringgit"),
    "IDR": ("USDIDR=X", True, "Indonesian rupiah"),
    "INR": ("USDINR=X", True, "Indian rupee"),
    "PHP": ("USDPHP=X", True, "Philippine peso"),
    # CEEMEA
    "PLN": ("USDPLN=X", True, "Polish zloty"),
    "HUF": ("USDHUF=X", True, "Hungarian forint"),
    "CZK": ("USDCZK=X", True, "Czech koruna"),
    "ZAR": ("USDZAR=X", True, "South African rand"),
    "TRY": ("USDTRY=X", True, "Turkish lira"),
    # LatAm
    "MXN": ("USDMXN=X", True, "Mexican peso"),
    "BRL": ("USDBRL=X", True, "Brazilian real"),
    "CLP": ("USDCLP=X", True, "Chilean peso"),
    "COP": ("USDCOP=X", True, "Colombian peso"),
}

BASE_CCY = "USD"

# Peer groups (a currency may appear in more than one group).
PEER_GROUPS: dict[str, list[str]] = {
    "G10 core / havens": ["EUR", "GBP", "CHF", "JPY"],
    "G10 commodity": ["AUD", "NZD", "CAD", "NOK"],
    "Scandies": ["NOK", "SEK"],
    "Asia EM": ["KRW", "TWD", "SGD", "THB", "MYR", "IDR", "INR", "PHP"],
    "CEEMEA": ["PLN", "HUF", "CZK", "ZAR", "TRY"],
    "LatAm": ["MXN", "BRL", "CLP", "COP"],
}

# The group used for each currency's "bloc vs own" decomposition.
PRIMARY_GROUP: dict[str, str] = {
    "EUR": "G10 core / havens", "GBP": "G10 core / havens", "CHF": "G10 core / havens", "JPY": "G10 core / havens",
    "AUD": "G10 commodity", "NZD": "G10 commodity", "CAD": "G10 commodity", "NOK": "G10 commodity",
    "SEK": "Scandies",
    "KRW": "Asia EM", "TWD": "Asia EM", "SGD": "Asia EM", "THB": "Asia EM",
    "MYR": "Asia EM", "IDR": "Asia EM", "INR": "Asia EM", "PHP": "Asia EM",
    "PLN": "CEEMEA", "HUF": "CEEMEA", "CZK": "CEEMEA", "ZAR": "CEEMEA", "TRY": "CEEMEA",
    "MXN": "LatAm", "BRL": "LatAm", "CLP": "LatAm", "COP": "LatAm",
}

# Currencies whose z-scores are unreliable (persistent one-way trend). Shown, but flagged.
TREND_FLAG: set[str] = {"TRY"}

# Session analysis is only reliable where Yahoo has liquid round-the-clock quotes.
SESSION_RELIABLE: set[str] = {"EUR", "GBP", "AUD", "NZD", "JPY", "CHF", "CAD", "SEK", "NOK", "SGD", "MXN", "ZAR"}

# ICE US Dollar Index: DXY = 50.14348112 * EURUSD^-0.576 * USDJPY^0.136 * GBPUSD^-0.119
#                              * USDCAD^0.091 * USDSEK^0.042 * USDCHF^0.036
# In Q terms (USD per unit) every factor becomes Q_c^(-w_c):  ln DXY = ln K - sum(w_c * ln Q_c)
DXY_CONSTANT = 50.14348112
DXY_WEIGHTS: dict[str, float] = {"EUR": 0.576, "JPY": 0.136, "GBP": 0.119, "CAD": 0.091, "SEK": 0.042, "CHF": 0.036}

# Pairs offered in the trade-ticket dropdown (any valid combination can also be uploaded via CSV).
TRADABLE_PAIRS: list[str] = [
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD", "USDSEK", "USDNOK",
    "EURGBP", "EURJPY", "EURCHF", "GBPJPY", "AUDJPY", "AUDNZD", "EURAUD", "EURNOK", "EURSEK",
    "USDSGD", "USDKRW", "USDINR", "USDMXN", "USDZAR", "USDBRL", "USDPLN", "USDTRY",
]

# Market quoting order: in a cross, the currency that comes first here is the base (EURGBP, AUDNZD, NOKSEK).
CCY_PRIORITY: list[str] = ["EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "NOK", "SEK", "JPY"]

# Data check (crosscheck.py): each currency is checked against a quoted cross, normally EURc. Yahoo's EUR
# crosses for some Asian currencies are unreliable, so these use SGD crosses instead. Measured over 500 days,
# typical (median) gap vs the USD legs: EURIDR 109 bps vs SGDIDR 2; EURKRW 5.6 (1 day in 10 > 42) vs SGDKRW 0.8;
# EURMYR 170 vs SGDMYR 2. Any other unreliable reference is caught automatically by crosscheck's reliability test.
CHECK_CROSS: dict[str, str] = {"IDR": "SGDIDR", "KRW": "SGDKRW", "MYR": "SGDMYR"}

def all_currencies() -> list[str]:
    return list(CURRENCIES)


def ticker_of(ccy: str) -> str:
    return CURRENCIES[ccy][0]


def is_known(ccy: str) -> bool:
    return ccy == BASE_CCY or ccy in CURRENCIES


def parse_pair(pair: str) -> tuple[str, str]:
    """'EURUSD' -> ('EUR', 'USD'). Raises ValueError for anything that is not two known currencies."""
    p = str(pair).upper().replace("/", "").replace("=X", "").strip()
    if len(p) != 6:
        raise ValueError(f"Pair '{pair}' must be 6 letters like EURUSD")
    base, quote = p[:3], p[3:]
    if base == quote:
        raise ValueError(f"Pair '{pair}' has the same base and quote currency")
    for c in (base, quote):
        if not is_known(c):
            raise ValueError(f"Currency '{c}' in pair '{pair}' is not in the universe")
    return base, quote


def g10_crosses() -> list[str]:
    """Every cross between two non-USD G10 currencies, in market quoting order (36 pairs)."""
    g10 = [c for c in CCY_PRIORITY if c != BASE_CCY]
    return [a + b for i, a in enumerate(g10) for b in g10[i + 1:]]


def price_decimals(pair: str) -> int:
    """Display precision: 2 dp for high-priced quotes (JPY, KRW...), 4 dp otherwise."""
    base, quote = parse_pair(pair)
    big = {"JPY", "KRW", "IDR", "INR", "PHP", "HUF", "CLP", "COP", "TWD", "THB", "CZK", "TRY"}
    return 2 if quote in big else 4
