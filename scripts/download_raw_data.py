import yfinance as yf
import pandas as pd
from tickers import tickers
from src.utils.paths import PROJECT_ROOT

def clean_yfinance_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = (
        pd.Index(df.columns)
        .map(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    df.columns.name = None
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.index.name = "timestamp"
    df = df.reset_index()

    df["ticker"] = ticker

    cols = ["ticker", "timestamp", "open", "high", "low", "close", "volume"]
    existing_cols = [col for col in cols if col in df.columns]
    df = df[existing_cols]

    return df


def download_many_tickers(
    tickers: list[str],
    start: str,
    end: str,
    interval: str = "1m",
) -> pd.DataFrame:
    all_dfs = []

    for ticker in tickers:
        print(f"Downloading {ticker}...")
        data = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )

        if data.empty:
            print(f"No data for {ticker}")
            continue

        df = clean_yfinance_df(data, ticker=ticker)
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No data downloaded.")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    return combined


df = download_many_tickers(
    tickers=tickers,
    start="2026-01-24",
    end="2026-03-24",
    interval="15m",
)

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

df.to_csv(f"{DATA_RAW_DIR}/market_data_15m_raw.csv", index=False)