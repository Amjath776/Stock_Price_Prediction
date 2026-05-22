"""
predict.py
----------
Entry point for the Stock Price Prediction pipeline.

This script orchestrates the full ML workflow in four sequential steps:
  1. Data Loading    — fetch historical OHLCV data via yfinance
  2. Preprocessing   — engineer features and build X / y
  3. Model Training  — split data and train a Random Forest Regressor
  4. Evaluation      — compute RMSE / MAE and print the final prediction

Run directly with:
    python predict.py

Logging is configured at INFO level by default so that all pipeline
stages emit progress messages to the console without affecting output.
To suppress debug-level messages, keep the level at INFO (default).
To see verbose debug details, change logging.INFO → logging.DEBUG below.
"""

import logging

# Configure root-level logging once here so all imported modules
# (data_loader, preprocessing, train_model, evaluate) inherit this setup
# without needing their own basicConfig calls.
logging.basicConfig(
    level=logging.INFO,                                      # Change to logging.DEBUG for verbose output
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Entry-point logger — identifies messages coming from this module
logger = logging.getLogger(__name__)

# Pipeline module imports — each handles one stage of the ML workflow
from data_loader import load_stock_data
from preprocessing import preprocess_data
from train_model import train_model
from evaluate import evaluate_model


def main():
    """
    Execute the full stock price prediction pipeline end-to-end.

    Pipeline stages
    ---------------
    1. load_stock_data   : Downloads 6 years of Apple (AAPL) daily OHLCV data.
    2. preprocess_data   : Engineers features (Return, MA_5, MA_10) and drops NaNs.
    3. train_model       : Trains a Random Forest Regressor on 80% of the data.
    4. evaluate_model    : Predicts on the last 20% and prints RMSE / MAE metrics.
    5. Final output      : Prints the last predicted price vs the actual closing price.

    The function wraps the entire pipeline in a try/except so that a single
    stage failure produces a clear error message without an unformatted traceback
    flooding the console.
    """
    logger.info("=" * 60)
    logger.info("[main] Stock Price Prediction Pipeline — STARTED")
    logger.info("=" * 60)

    try:
        # --- Step 1: Data Loading ---
        # Download historical AAPL prices from Yahoo Finance (2018–2024)
        logger.info("[main] Step 1/4 — Loading stock data...")
        df = load_stock_data("AAPL")
        logger.info("[main] Step 1/4 — Data loading complete. Rows fetched: %d", len(df))

        # --- Step 2: Preprocessing ---
        # Engineer features and split into feature matrix X and target y
        logger.info("[main] Step 2/4 — Preprocessing data...")
        X, y = preprocess_data(df)
        logger.info("[main] Step 2/4 — Preprocessing complete. X shape: %s | y length: %d",
                    X.shape, len(y))

        # --- Step 3: Model Training ---
        # Train Random Forest on the first 80% of the time series
        logger.info("[main] Step 3/4 — Training the model...")
        model, X_test, y_test = train_model(X, y)
        logger.info("[main] Step 3/4 — Training complete.")

        # --- Step 4: Evaluation ---
        # Evaluate on the held-out last 20% and collect raw predictions
        logger.info("[main] Step 4/4 — Evaluating model performance...")
        predictions = evaluate_model(model, X_test, y_test)
        logger.info("[main] Step 4/4 — Evaluation complete.")

        # --- Final Output ---
        # Compare the most recent prediction against the actual closing price
        # (last element of the test set, i.e., the most recent trading day)
        print("Predicted price:", predictions[-1])
        print("Actual price:", y_test[-1])
        logger.info("[main] Final prediction — Predicted: %.4f | Actual: %.4f",
                    predictions[-1], y_test[-1])

        # Log the absolute dollar error as a quick sanity check
        abs_error = abs(predictions[-1] - y_test[-1])
        logger.info("[main] Absolute error on final prediction: $%.4f", abs_error)

    except Exception as e:
        # Catch any unhandled exception from any pipeline stage and log a
        # clear, actionable error message before the program exits
        logger.error("[main] PIPELINE FAILED. Reason: %s", e)
        logger.error("[main] Check the logs above for the exact stage that failed.")
        raise

    logger.info("=" * 60)
    logger.info("[main] Stock Price Prediction Pipeline — COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    # This block runs only when the script is executed directly (not imported),
    # making the pipeline both a standalone script and an importable module.
    main()
