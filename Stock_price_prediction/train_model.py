"""
train_model.py
--------------
Handles splitting the prepared dataset into training and test sets, and
training a Random Forest Regressor model to predict stock closing prices.

Model details
-------------
- Algorithm  : RandomForestRegressor (ensemble of decision trees)
- n_estimators : 100 trees — balances accuracy and training speed
- random_state  : 42  — ensures reproducible results across runs
- shuffle=False : Preserves the natural time ordering of the data, which
                  is critical for time-series to prevent look-ahead bias
"""

import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Module-level logger — consistent with other modules in this pipeline
logger = logging.getLogger(__name__)

# --- Hyperparameter constants ---
# Defining these at module level makes them easy to spot and adjust without
# touching the function body. Change here and the entire file updates.
TEST_SIZE    = 0.2   # Fraction of data held out for testing (last 20% by time)
N_ESTIMATORS = 100   # Number of trees in the Random Forest ensemble
RANDOM_STATE = 42    # Seed for reproducibility across runs


def train_model(X, y):
    """
    Split the dataset chronologically and train a Random Forest Regressor.

    The test split is kept at the *end* of the time series (shuffle=False)
    to simulate a realistic scenario where the model predicts future prices
    based on past data only — avoiding look-ahead bias.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix produced by `preprocess_data()`.
        Expected columns: Open, High, Low, Volume, Return, MA_5, MA_10.
    y : numpy.ndarray
        Target vector of closing prices produced by `preprocess_data()`.

    Returns
    -------
    model : sklearn.ensemble.RandomForestRegressor
        The trained Random Forest model, ready for evaluation or prediction.
    X_test : pandas.DataFrame
        The held-out feature set (last 20% of the data).
    y_test : numpy.ndarray
        The held-out target values (last 20% of the data).

    Raises
    ------
    ValueError
        If X and y have incompatible shapes or are empty.
    Exception
        Any unexpected training error is logged and re-raised.
    """
    logger.info("[train_model] Starting model training. Total samples: %d | Features: %d",
                len(X), X.shape[1])

    try:
        # --- Train / Test Split ---
        # Use the last TEST_SIZE fraction as the test set (shuffle=False preserves
        # temporal order — essential for any time-series model)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, shuffle=False
        )
        logger.info("[train_model] Data split — Training samples: %d | Test samples: %d",
                    len(X_train), len(X_test))

        # --- Model Instantiation ---
        # RandomForestRegressor: an ensemble of N_ESTIMATORS decision trees that each
        # vote on the predicted price; their average becomes the final prediction.
        # RANDOM_STATE ensures the same trees are built every run.
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE
        )
        logger.debug("[train_model] RandomForestRegressor instantiated with "
                     "n_estimators=%d, random_state=%d.", N_ESTIMATORS, RANDOM_STATE)

        # --- Training ---
        # model.fit() builds all 100 trees on the training portion of the data
        logger.info("[train_model] Fitting model on training data — this may take a moment...")
        model.fit(X_train, y_train)
        logger.info("[train_model] Model training complete.")

        return model, X_test, y_test

    except ValueError as e:
        logger.error("[train_model] ERROR: Data shape mismatch or empty dataset. "
                     "X shape: %s | y shape: %s | Reason: %s", X.shape, y.shape, e)
        raise
    except Exception as e:
        logger.error("[train_model] ERROR: Unexpected failure during model training. Reason: %s", e)
        raise
