"""
evaluate.py
-----------
Evaluates the trained stock price prediction model using standard regression
metrics and returns the raw predictions for further use.

Metrics reported
----------------
- RMSE (Root Mean Squared Error) : Penalises large prediction errors more
  heavily than MAE. Expressed in the same unit as the stock price (USD).
- MAE  (Mean Absolute Error)     : The average absolute difference between
  predicted and actual closing prices. Easy to interpret as a dollar error.
"""

import logging
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Module-level logger — consistent with other modules in this pipeline
logger = logging.getLogger(__name__)

# --- Evaluation thresholds ---
# Named constants for RMSE quality bands; adjust here to change what counts
# as "excellent" or "acceptable" without touching the if/elif logic below.
RMSE_EXCELLENT   = 5    # RMSE below this value → model quality is excellent
RMSE_ACCEPTABLE  = 20   # RMSE below this value → model quality is acceptable


def evaluate_model(model, X_test, y_test):
    """
    Generate predictions on the test set and compute regression metrics.

    The function prints RMSE and MAE to stdout (original behaviour preserved)
    and also logs them at INFO level for developer-friendly debugging.

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestRegressor
        A trained model returned by `train_model()`.
    X_test : pandas.DataFrame
        The held-out feature set (last 20% of the data by time).
    y_test : numpy.ndarray
        The true closing prices for the test period.

    Returns
    -------
    predictions : numpy.ndarray
        Array of predicted closing prices for each row in X_test.

    Raises
    ------
    Exception
        Any failure during prediction or metric computation is logged with
        context before the exception propagates to the caller.
    """
    logger.info("[evaluate] Running model evaluation on %d test samples.", len(X_test))

    try:
        # Generate price predictions for the held-out test set
        predictions = model.predict(X_test)
        logger.debug("[evaluate] Predictions generated. Sample (last 5): %s", predictions[-5:])

        # --- RMSE ---
        # Square root of the average squared difference between predicted and
        # actual prices. Penalises large errors more than MAE.
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        # --- MAE ---
        # Simple average of absolute errors — directly interpretable as the
        # average dollar error per prediction.
        mae = mean_absolute_error(y_test, predictions)

        # Print metrics to stdout (original behaviour — kept unchanged)
        print("RMSE:", rmse)
        print("MAE:", mae)

        # Additionally log metrics at INFO level for developer visibility
        logger.info("[evaluate] Evaluation complete — RMSE: %.4f | MAE: %.4f", rmse, mae)

        # Qualitative hint in logs — uses named threshold constants defined above
        if rmse < RMSE_EXCELLENT:
            logger.info("[evaluate] Model quality: EXCELLENT (RMSE < %d)", RMSE_EXCELLENT)
        elif rmse < RMSE_ACCEPTABLE:
            logger.info("[evaluate] Model quality: ACCEPTABLE (RMSE between %d and %d)",
                        RMSE_EXCELLENT, RMSE_ACCEPTABLE)
        else:
            logger.warning("[evaluate] Model quality: POOR (RMSE >= %d). "
                           "Consider tuning hyperparameters or adding more features.", RMSE_ACCEPTABLE)

        return predictions

    except Exception as e:
        logger.error("[evaluate] ERROR: Failure during model evaluation. Reason: %s", e)
        raise
