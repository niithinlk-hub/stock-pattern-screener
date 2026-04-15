"""
modules/data.py
Data fetching layer: stock lists + OHLCV downloads via yfinance.
All heavy network calls are cached with st.cache_data.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded ticker lists  (used as primary source for NASDAQ,
#                          and as reliable fallback for S&P 500)
# ─────────────────────────────────────────────────────────────────────────────

# Full S&P 500 constituent list (~503 tickers, as of early 2025)
_SP500_TICKERS = [
    "A", "AAL", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE",
    "ADI", "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG",
    "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR",
    "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN", "ANSS", "AON", "AOS",
    "APA", "APD", "APH", "APTV", "ARE", "ATO", "AVB", "AVGO", "AVY",
    "AWK", "AXON", "AXP", "AZO",
    "BA", "BAC", "BALL", "BAX", "BBY", "BDX", "BEN", "BF-B", "BIIB",
    "BIO", "BKR", "BLK", "BLDR", "BMY", "BR", "BRK-B", "BRO", "BX", "BG",
    "C", "CAG", "CAH", "CAT", "CB", "CBOE", "CBRE", "CCL", "CDNS", "CDW",
    "CE", "CEG", "CF", "CFG", "CHD", "CHTR", "CI", "CINF", "CL", "CLX",
    "CMA", "CME", "CMG", "CMI", "CMS", "CNC", "COF", "COO", "COP", "COR",
    "COST", "CPAY", "CPB", "CPRT", "CRL", "CRM", "CSCO", "CSGP", "CSX",
    "CTLT", "CTSH", "CTVA", "CVS", "CVX", "CZR",
    "D", "DAL", "DD", "DE", "DDOG", "DFS", "DG", "DHI", "DHR", "DIS",
    "DLTR", "DLR", "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK",
    "DVA", "DVN", "DXCM",
    "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV", "EMN",
    "EMR", "ENPH", "EOG", "EPAM", "EQR", "EQIX", "EQT", "ESS", "ETN",
    "ETR", "ETSY", "EVRG", "EW", "EXC", "EXPE", "EXPD", "EXR",
    "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FE", "FICO", "FI", "FIS",
    "FITB", "FSLR", "FMC", "FOXA", "FOX", "FRT", "FTNT", "FTV",
    "GD", "GE", "GEHC", "GEN", "GILD", "GIS", "GM", "GOOGL", "GOOG",
    "GPC", "GS", "GWW",
    "HAL", "HAS", "HCA", "HD", "HES", "HIG", "HII", "HLT", "HOLX", "HON",
    "HPE", "HPQ", "HRL", "HST", "HSIC", "HSY", "HUBB", "HUM", "HBAN", "HWM",
    "IBM", "ICE", "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTC", "INTU",
    "INVH", "IP", "IPG", "IQV", "IR", "IRM", "ISRG", "ITW", "IVZ",
    "J", "JBL", "JBHT", "JCI", "JKHY", "JNJ", "JPM", "JNPR",
    "K", "KEYS", "KEY", "KLAC", "KHC", "KDP", "KIM", "KMB", "KMI",
    "KMX", "KO", "KR", "KVUE",
    "L", "LEN", "LH", "LHX", "LIN", "LKQ", "LDOS", "LLY", "LMT", "LNC",
    "LOW", "LRCX", "LUV", "LULU", "LVS", "LW", "LYB", "LYV",
    "MA", "MAA", "MAR", "MAS", "MCD", "MCK", "MCO", "MCHP", "MDT", "MET",
    "MHK", "MNST", "MKC", "MKTX", "MLM", "MO", "MOH", "MOS", "MPWR",
    "MPC", "MRO", "MRNA", "MS", "MSCI", "MSFT", "MSI", "MTB", "MTD", "MU",
    "NDAQ", "NEM", "NEE", "NFLX", "NI", "NKE", "NDSN", "NSC", "NTAP",
    "NTRS", "NUE", "NVDA", "NVR", "NWSA", "NWS", "NXPI", "NCLH", "NRG",
    "O", "ODFL", "OGN", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY",
    "PCAR", "PANW", "PAYC", "PAYX", "PCG", "PEG", "PEP", "PFE", "PFG",
    "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PM", "PNC", "PNR", "PNW",
    "PODD", "POOL", "PPG", "PPL", "PRU", "PSA", "PSX", "PWR", "PYPL",
    "QCOM", "QRVO",
    "RL", "RJF", "RCL", "REG", "REGN", "RF", "RVTY", "RMD", "ROK", "ROL",
    "ROP", "ROST", "RSG", "RTX",
    "SBAC", "SBUX", "SEE", "SHW", "SJM", "SLB", "SMCI", "SNA", "SNPS",
    "SOLV", "SO", "SPG", "SPGI", "SRE", "STT", "STX", "STLD", "STE",
    "STZ", "SWKS", "SWK", "SYF", "SYK", "SYY",
    "T", "TAP", "TDG", "TDY", "TEL", "TER", "TFC", "TFX", "TGT", "TJX",
    "TMUS", "TMO", "TPR", "TRGP", "TRMB", "TSCO", "TSLA", "TT", "TTWO",
    "TXN", "TYL", "TSN",
    "UAL", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB",
    "V", "VLO", "VMC", "VTR", "VLTO", "VICI", "VRSK", "VRSN", "VRTX",
    "VTRS", "VZ",
    "WAB", "WAT", "WBA", "WBD", "WDC", "WEC", "WELL", "WFC", "WHR",
    "WM", "WMB", "WMT", "WRB", "WST", "WTW", "WYNN", "WY",
    "XEL", "XOM", "XYL", "XRAY",
    "YUM",
    "ZBRA", "ZBH", "ZTS",
]

# NASDAQ 100 core + additional large NASDAQ-listed technology & growth stocks
_NASDAQ_TICKERS = [
    # ── NASDAQ 100 components ──────────────────────────────────────────────────
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOGL", "GOOG", "AVGO",
    "COST", "NFLX", "TMUS", "AMD", "AMGN", "PEP", "QCOM", "CSCO", "INTU",
    "TXN", "CMCSA", "ISRG", "HON", "BKNG", "SBUX", "ADP", "VRTX", "GILD",
    "MU", "ADI", "LRCX", "PANW", "REGN", "MDLZ", "SNPS", "CDNS", "KLAC",
    "MELI", "ASML", "KDP", "INTC", "CSX", "ABNB", "MRVL", "MAR", "PYPL",
    "ORLY", "ADSK", "MNST", "CTAS", "FTNT", "NXPI", "WDAY", "CPRT", "ROP",
    "IDXX", "AZN", "PCAR", "DDOG", "CHTR", "CRWD", "FAST", "PAYX", "MCHP",
    "EA", "ODFL", "BKR", "BIIB", "LULU", "VRSK", "FANG", "ANSS", "CTSH",
    "DXCM", "TTWO", "ILMN", "TEAM", "DLTR", "MRNA", "ZS", "ALGN", "CDW",
    "ENPH", "ON", "GEHC", "ROST", "EXC", "EBAY",
    # ── Semiconductors & Hardware ──────────────────────────────────────────────
    "SMCI", "AMAT", "TER", "MPWR", "ONTO", "ACLS", "WOLF", "COHU",
    "NTAP", "WDC", "STX", "PSTG", "DELL", "HPE",
    # ── Cloud, SaaS & Enterprise Software ─────────────────────────────────────
    "CRM", "NOW", "ADBE", "ORCL", "VEEV", "HUBS", "MDB", "OKTA", "SNOW",
    "ZM", "DOCU", "BILL", "GTLB", "DOCN", "BOX", "DDOG",
    # ── Cybersecurity ─────────────────────────────────────────────────────────
    "NET", "S", "CYBR", "AKAM", "TENB", "QLYS",
    # ── Fintech & Payments ────────────────────────────────────────────────────
    "SQ", "COIN", "HOOD", "SOFI", "AFRM", "UPST", "NDAQ",
    # ── E-commerce, Media & Consumer Tech ────────────────────────────────────
    "SHOP", "ETSY", "EBAY", "ROKU", "SPOT", "PINS", "SNAP", "RBLX",
    "TTD", "DASH", "UBER", "LYFT",
    # ── Electric Vehicles ─────────────────────────────────────────────────────
    "RIVN", "LCID", "NIO", "LI", "XPEV",
    # ── Biotech & Life Sciences ───────────────────────────────────────────────
    "ALNY", "BMRN", "SGEN", "EXEL", "RARE", "ACAD", "FOLD",
    # ── China Tech (NASDAQ-listed ADRs) ───────────────────────────────────────
    "BIDU", "JD", "PDD", "NTES",
    # ── AI, Data & Analytics ──────────────────────────────────────────────────
    "PLTR", "APP", "MSTR", "AI", "BBAI",
]

# ── NSE request headers to reduce blocking probability ────────────────────────
_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}

# ── Nifty 50 fallback symbols (no .NS suffix — added later) ───────────────────
_NIFTY50_FALLBACK = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR",
    "ICICIBANK", "KOTAKBANK", "BHARTIARTL", "ITC", "BAJFINANCE",
    "SBIN", "LT", "AXISBANK", "ASIANPAINT", "DMART",
    "MARUTI", "TITAN", "WIPRO", "ULTRACEMCO", "POWERGRID",
    "NTPC", "SUNPHARMA", "HCLTECH", "ONGC", "TECHM",
    "ADANIENT", "ADANIPORTS", "COALINDIA", "JSWSTEEL", "TATAMOTORS",
    "INDUSINDBK", "TATASTEEL", "BPCL", "GRASIM", "M&M",
    "CIPLA", "EICHERMOT", "HEROMOTOCO", "DRREDDY", "NESTLEIND",
    "APOLLOHOSP", "BAJAJ-AUTO", "DIVISLAB", "BRITANNIA", "BAJAJFINSV",
    "SBILIFE", "HDFCLIFE", "UPL", "TATACONSUM", "HINDALCO",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: robust column finder (case-insensitive, strips BOM/whitespace)
# ─────────────────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, *keywords: str) -> str | None:
    """Return the first column name that contains any of the given keywords."""
    clean = {c: c.strip().lstrip("\ufeff").lower() for c in df.columns}
    for kw in keywords:
        for orig, norm in clean.items():
            if kw in norm:
                return orig
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_sp500_tickers() -> tuple[list[str], pd.DataFrame]:
    """
    Fetch the S&P 500 constituent list (with names & sectors) from GitHub CSV.
    Always returns all ~503 tickers — the hardcoded list is the reliable base;
    the GitHub fetch only enriches names/sectors.

    Returns
    -------
    tickers : list[str]   yfinance-compatible (BRK.B → BRK-B)
    info_df : pd.DataFrame   columns: ticker, name, sector
    """
    csv_url = (
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies"
        "/main/data/constituents.csv"
    )
    try:
        resp = requests.get(csv_url, timeout=12)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))

        sym_col    = _find_col(df, "symbol")
        name_col   = _find_col(df, "name", "security", "company")
        sector_col = _find_col(df, "sector", "gics sector", "industry")

        if sym_col is None:
            raise ValueError(f"No symbol column found. Columns: {list(df.columns)}")

        symbols = df[sym_col].str.strip().str.replace(".", "-", regex=False).tolist()
        names   = df[name_col].tolist()   if name_col   else symbols
        sectors = df[sector_col].tolist() if sector_col else ["N/A"] * len(symbols)

        info_df = pd.DataFrame({"ticker": symbols, "name": names, "sector": sectors})
        return symbols, info_df

    except Exception as exc:
        st.warning(f"Could not fetch S&P 500 names from GitHub ({exc}). Using hardcoded list.")

    # Fallback — full hardcoded list (symbols only, no names)
    info_df = pd.DataFrame({
        "ticker": _SP500_TICKERS,
        "name":   _SP500_TICKERS,
        "sector": ["N/A"] * len(_SP500_TICKERS),
    })
    return _SP500_TICKERS, info_df


@st.cache_data(ttl=3600, show_spinner=False)
def get_nasdaq_tickers() -> tuple[list[str], pd.DataFrame]:
    """
    Return the curated NASDAQ 100 + large-cap NASDAQ technology stock list.
    No external fetch needed — the list is hardcoded for reliability.

    Returns
    -------
    tickers : list[str]   yfinance-compatible ticker symbols
    info_df : pd.DataFrame   columns: ticker, name, sector
    """
    info_df = pd.DataFrame({
        "ticker": _NASDAQ_TICKERS,
        "name":   _NASDAQ_TICKERS,   # names enriched at runtime if needed
        "sector": ["Technology"] * len(_NASDAQ_TICKERS),
    })
    return _NASDAQ_TICKERS, info_df


@st.cache_data(ttl=3600, show_spinner=False)
def get_nifty500_tickers() -> tuple[list[str], pd.DataFrame]:
    """
    Fetch the Nifty 500 constituent list from NSE archives.
    Falls back to a hardcoded Nifty 50 list if NSE blocks the request.

    Returns
    -------
    tickers : list[str]   yfinance-compatible NSE symbols with '.NS' suffix.
    info_df : pd.DataFrame   columns: ticker, name, sector.
    """
    nse_url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        resp = requests.get(nse_url, headers=_NSE_HEADERS, timeout=12)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))

        sym_col    = _find_col(df, "symbol")
        name_col   = _find_col(df, "company", "name")
        sector_col = _find_col(df, "industry", "sector")

        if sym_col is None:
            raise ValueError(f"No symbol column found. Columns: {list(df.columns)}")

        symbols = df[sym_col].str.strip().tolist()
        names   = df[name_col].tolist()   if name_col   else symbols
        sectors = df[sector_col].tolist() if sector_col else ["N/A"] * len(symbols)
        tickers = [f"{s}.NS" for s in symbols]

        info_df = pd.DataFrame({"ticker": tickers, "name": names, "sector": sectors})
        return tickers, info_df

    except Exception as exc:
        st.warning(
            f"Could not fetch Nifty 500 from NSE ({exc}). "
            "Using Nifty 50 fallback list instead."
        )
        tickers = [f"{s}.NS" for s in _NIFTY50_FALLBACK]
        info_df = pd.DataFrame({
            "ticker": tickers,
            "name":   _NIFTY50_FALLBACK,
            "sector": ["N/A"] * len(tickers),
        })
        return tickers, info_df


@st.cache_data(ttl=14400, show_spinner=False)
def download_batch(
    tickers: tuple,   # tuple for cache hashability
    period: str = "1y",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """
    Batch-download OHLCV data for a list of tickers using yfinance.

    Parameters
    ----------
    tickers : tuple[str]
        Must be a tuple so Streamlit can hash it for caching.
    period : str
        yfinance period string, e.g. '6mo', '1y', '2y', '3y'.
    interval : str
        yfinance interval string, e.g. '1d', '1wk'.

    Returns
    -------
    dict mapping ticker → pd.DataFrame (OHLCV, auto-adjusted).
    Tickers with < 60 valid rows are excluded.
    """
    tickers_list = list(tickers)
    if not tickers_list:
        return {}

    try:
        raw = yf.download(
            tickers=tickers_list,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
    except Exception as exc:
        st.error(f"yfinance download error: {exc}")
        return {}

    result: dict[str, pd.DataFrame] = {}

    if len(tickers_list) == 1:
        # Single ticker → flat DataFrame (no MultiIndex)
        df = raw.dropna(how="all")
        if len(df) >= 60:
            result[tickers_list[0]] = df
        return result

    # Multiple tickers → MultiIndex columns
    for ticker in tickers_list:
        try:
            df = raw[ticker].dropna(how="all")
            if len(df) >= 60:
                result[ticker] = df
        except (KeyError, TypeError):
            pass

    return result
