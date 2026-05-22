"""
data_loader.py
--------------
Responsible for fetching historical stock price data from Yahoo Finance
using the yfinance library. This module serves as the data ingestion
layer of the Stock Price Prediction pipeline.
"""

import logging
import yfinance as yf

# Configure a module-level logger so all messages are clearly attributed
# to this file in the console output.
logger = logging.getLogger(__name__)


def load_stock_data(ticker="AAPL", start="2018-01-01", end="2024-01-01"):
    """
    Download historical OHLCV (Open, High, Low, Close, Volume) data for a
    given stock ticker from Yahoo Finance.

    Parameters
    ----------
    ticker : str, optional
        The stock ticker symbol to download (default: "AAPL" for Apple Inc.).
    start : str, optional
        The start date for the historical data in "YYYY-MM-DD" format
        (default: "2018-01-01").
    end : str, optional
        The end date for the historical data in "YYYY-MM-DD" format
        (default: "2024-01-01").

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the downloaded stock data with the Date
        column reset as a regular column (not the index).

    Raises
    ------
    Exception
        Propagates any exception raised by yfinance (e.g., network errors,
        invalid ticker symbols) after logging a descriptive error message.
    """
    logger.info("[data_loader] Starting data download for ticker='%s' | range: %s → %s", ticker, start, end)

    try:
        # Fetch raw OHLCV data from Yahoo Finance for the requested date range
        df = yf.download(ticker, start=start, end=end, progress=False)

        # Guard against an empty download (e.g., invalid ticker or no data
        # available for the requested range) before further processing
        if df.empty:
            logger.warning("[data_loader] WARNING: No data returned for ticker='%s'. "
                           "Check the ticker symbol and date range.", ticker)
        else:
            logger.info("[data_loader] Successfully downloaded %d rows for '%s'.", len(df), ticker)

        # Reset index so 'Date' becomes a regular column rather than the index,
        # making it easier to reference in downstream preprocessing steps
        df = df.reset_index()
        logger.debug("[data_loader] DataFrame shape after reset_index: %s", df.shape)

        return df

    except Exception as e:
        # Log a clear, descriptive error so developers can quickly identify
        # whether the failure is a network issue, bad ticker, or API change
        logger.error("[data_loader] ERROR: Failed to download data for ticker='%s'. "
                     "Reason: %s", ticker, e)
        raise
