"""
preprocessing.py
----------------
Transforms raw stock price data into a feature matrix (X) and target
vector (y) suitable for training a machine learning model.

Feature engineering steps performed here:
  - Daily percentage return  (momentum signal)
  - 5-day moving average     (short-term trend)
  - 10-day moving average    (medium-term trend)
  - Rows with NaN values are dropped (result of rolling calculations)
"""

import logging

# Module-level logger — consistent with other modules in this pipeline
logger = logging.getLogger(__name__)

# Ordered list of feature columns used to build the model input matrix.
# Centralising this here means adding or removing a feature is a single
# one-line change, with no risk of the list drifting out of sync elsewhere.
FEATURE_COLS = ["Open", "High", "Low", "Volume", "Return", "MA_5", "MA_10"]


def preprocess_data(df):
    """
    Engineer features from raw OHLCV stock data and split it into a
    feature matrix X and a target vector y.

    Features used
    -------------
    - Open    : Opening price of the trading session
    - High    : Highest price reached during the session
    - Low     : Lowest price reached during the session
    - Volume  : Number of shares traded
    - Return  : Daily percentage change in closing price (momentum indicator)
    - MA_5    : 5-day rolling mean of closing price (short-term trend)
    - MA_10   : 10-day rolling mean of closing price (medium-term trend)

    Target
    ------
    - Close   : The closing price — this is what the model learns to predict.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw stock DataFrame returned by `load_stock_data()`. Must contain
        at least the columns: Open, High, Low, Close, Volume.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix with shape (n_samples, 7).
    y : numpy.ndarray
        1-D target array of closing prices with shape (n_samples,).

    Raises
    ------
    KeyError
        If any expected column (e.g., 'Close') is missing from the DataFrame,
        a descriptive error is logged before the exception propagates.
    Exception
        Any other unexpected errors during preprocessing are logged and
        re-raised so callers can handle them appropriately.
    """
    logger.info("[preprocessing] Starting feature engineering. Input shape: %s", df.shape)

    try:
        # Work on a copy to avoid mutating the original DataFrame, which
        # could cause hard-to-trace bugs if the caller reuses the object
        df = df.copy()
        logger.debug("[preprocessing] Working on a copy of the DataFrame to preserve the original.")

        # --- Feature 1: Daily Return ---
        # pct_change() computes (Close_today - Close_yesterday) / Close_yesterday
        # This captures short-term price momentum
        df["Return"] = df["Close"].pct_change()
        logger.debug("[preprocessing] Computed daily percentage return ('Return' column).")

        # --- Feature 2 & 3: Moving Averages ---
        # Rolling means smooth out short-term price noise and reveal trends
        df["MA_5"]  = df["Close"].rolling(5).mean()   # 5-day (short-term trend)
        df["MA_10"] = df["Close"].rolling(10).mean()  # 10-day (medium-term trend)
        logger.debug("[preprocessing] Computed MA_5 and MA_10 rolling averages.")

        # Drop rows that contain NaN values introduced by pct_change() (first row)
        # and the rolling windows (first 4 rows for MA_5, first 9 for MA_10).
        # This ensures a clean, aligned dataset for the model.
        rows_before = len(df)
        df.dropna(inplace=True)
        rows_dropped = rows_before - len(df)
        logger.info("[preprocessing] Dropped %d NaN rows (from rolling calculations). "
                    "Remaining rows: %d", rows_dropped, len(df))

        # --- Build feature matrix X and target vector y ---
        # Use the module-level FEATURE_COLS constant — single source of truth
        X = df[FEATURE_COLS]

        # ravel() converts to a flat 1-D array, required by scikit-learn regressors
        y = df["Close"].values.ravel()

        logger.info("[preprocessing] Feature engineering complete. X shape: %s | y shape: %s",
                    X.shape, y.shape)

        return X, y

    except KeyError as e:
        logger.error("[preprocessing] ERROR: Expected column not found in DataFrame: %s. "
                     "Ensure the raw data contains Open, High, Low, Close, Volume columns.", e)
        raise
    except Exception as e:
        logger.error("[preprocessing] ERROR: Unexpected failure during preprocessing. Reason: %s", e)
        raise
