import time                                     
import numpy as np
import pandas as pd
import random
import itertools
from sklearn.metrics import mean_absolute_error
from config import TUNING_EPOCHS  # For tuning runs

def create_forecast_pairs(series, window_size, horizon=1):
    """Generate forecasting (X, y) pairs from a 1D numpy array."""
    X, y = [], []
    n = len(series)
    for i in range(n - window_size - horizon + 1):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size + horizon - 1])
    return np.array(X), np.array(y)

def forecast_evaluation(imputed_df, train_series_df, test_series_df, window_size):
    import xgboost as xgb
    forecast_results = {}
    for col in imputed_df.columns:
        train_series = train_series_df[col].values
        X_train, y_train = create_forecast_pairs(train_series, window_size)
        model = xgb.XGBRegressor(
            objective='reg:squarederror', n_estimators=100,
            random_state=42, verbosity=0
        )
        model.fit(X_train, y_train)
        test_series = imputed_df[col].values
        X_test, y_test = create_forecast_pairs(test_series, window_size)
        y_pred = model.predict(X_test)
        forecast_results[col] = mean_absolute_error(y_test, y_pred)
    avg_mae = np.mean(list(forecast_results.values()))
    return avg_mae, forecast_results

def forecast_transformer_eval(model, test_series, window_size, horizon=1, train_mean=None, train_std=None):
    X_test, y_test = create_forecast_pairs(test_series, window_size, horizon)
    predictions = model.predict({"X": X_test})
    from data_utils import process_prediction_result
    preds = process_prediction_result(predictions).squeeze(axis=1)
    y_test_denorm = y_test * train_std.values + train_mean.values
    preds_denorm = preds * train_std.values + train_mean.values
    return mean_absolute_error(y_test_denorm, preds_denorm)

def evaluate_imputation_full(original, simulated, method="mean", train_df=None, train_std=None, train_mean=None, eps=1e-6):
    imputed = simulated.copy()
    mask = simulated.isnull()
    if method == "mean":
        for col in simulated.columns:
            imputed[col] = imputed[col].fillna(train_df[col].mean())
    elif method == "median":
        for col in simulated.columns:
            imputed[col] = imputed[col].fillna(train_df[col].median())
    elif method == "linear":
        imputed = imputed.interpolate(method="linear", limit_direction="both")
    else:
        raise ValueError("Method must be 'mean','median', or 'linear'.")

    imputed_denorm = imputed * train_std + train_mean
    total_error, total_rel_error, total_count = 0, 0, 0

    for col in simulated.columns:
        missing_idx = mask[col]
        if missing_idx.sum() > 0:
            errors = np.abs(original.loc[missing_idx, col] - imputed_denorm.loc[missing_idx, col])
            total_error     += errors.sum()
            total_rel_error += (errors / (np.abs(original.loc[missing_idx, col]) + eps)).sum()
            total_count     += missing_idx.sum()

    mae = total_error / total_count if total_count > 0 else np.nan
    mre = total_rel_error / total_count if total_count > 0 else np.nan
    return mae, mre

def evaluate_imputation_windowed(original_windows, simulated_windows, model, train_std, train_mean, eps=1e-6):
    """
    Returns: (mae, mre, inference_time)
    where inference_time = time spent in model.predict + process_prediction_result
    """
    mask = np.isnan(simulated_windows)

    # ── measure imputation/inference only ─────────────────────────────
    t0 = time.perf_counter()
    result = model.predict({"X": simulated_windows})
    from data_utils import process_prediction_result
    out = process_prediction_result(result)
    t1 = time.perf_counter()
    inference_time = t1 - t0
    # ─────────────────────────────────────────────────────────────────

    if out.ndim == 4:
        out = out.mean(axis=1)
    expected_features = original_windows.shape[-1]
    if out.shape[-1] != expected_features:
        out = out[..., :expected_features]

    imputed_denorm = out * train_std.values + train_mean.values

    total_error, total_rel_error, total_count = 0, 0, 0
    N, L, F = imputed_denorm.shape

    for i in range(N):
        for j in range(L):
            for k in range(F):
                if mask[i, j, k]:
                    diff = abs(original_windows[i, j, k] - imputed_denorm[i, j, k])
                    total_error     += diff
                    total_rel_error += diff / (abs(original_windows[i, j, k]) + eps)
                    total_count     += 1

    mae = total_error / total_count if total_count > 0 else np.nan
    mre = total_rel_error / total_count if total_count > 0 else np.nan

    return mae, mre, inference_time


def advanced_tune_model(
    model_class,
    param_grid,
    train_windows,
    val_windows,
    model_name,
    common_params,
    n_iter=10,
    missing_rate=0.3,
    train_std=None,
    train_mean=None
):
    """
    Random-search hyperparameter tuning for imputation models.

    Returns
    -------
    best_params : dict
        The hyperparameter setting that achieved the lowest MAE.
    results : List[dict]
        A list of all tried settings, each with its recorded MAE.
    """
    # 1. Expand grid and sample candidates
    keys, values = zip(*param_grid.items())
    all_candidates = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    if len(all_candidates) > n_iter:
        candidates = random.sample(all_candidates, n_iter)
    else:
        candidates = all_candidates

    results = []
    best_mae = float("inf")
    best_params = None

    # 2. Prepare a “noisy” validation set with artificial missingness
    val_missing = val_windows.copy()
    mask = np.random.rand(*val_missing.shape) < missing_rate
    val_missing[mask] = np.nan

    # 3. Loop through candidates
    for candidate in candidates:
        try:
            # 3a. Instantiate with both common and candidate‐specific arguments
            model = model_class(
                n_steps=common_params["window_size"],
                n_features=train_windows.shape[-1],
                **candidate,
                batch_size=common_params.get("batch_size"),
                epochs=TUNING_EPOCHS,
                patience=common_params.get("patience"),
                num_workers=common_params.get("num_workers"),
                device=common_params.get("device"),
                saving_path=None,
                model_saving_strategy=None
            )

            # 3b. Fit on the clean training windows
            model.fit({"X": train_windows})

            # 3c. Evaluate on the corrupted validation windows
            mae, _ , _ = evaluate_imputation_windowed(
                val_windows,       # "ground truth" windows
                val_missing,       # with injected missingness
                model,
                train_std,
                train_mean
            )

            # Record this trial
            trial = candidate.copy()
            trial["MAE"] = mae
            results.append(trial)

            # Update best if improved
            if mae < best_mae:
                best_mae = mae
                best_params = candidate

        except Exception as e:
            print(f"Error tuning {model_name} with candidate {candidate}: {e}")

    return best_params, results


# def advanced_tune_model_(model_class, param_grid, train_windows, val_windows, model_name, common_params, n_iter=10, missing_rate=0.3, train_std=None, train_mean=None):
#     """
#     Perform advanced hyperparameter tuning via random search.
#     Uses TUNING_EPOCHS for the tuning runs.
#     """
#     print(f"Tuning {model_name} with parameter grid: {param_grid}")
#     keys, values = zip(*param_grid.items())
#     all_candidates = [dict(zip(keys, v)) for v in itertools.product(*values)]
#     candidates = random.sample(all_candidates, n_iter) if len(all_candidates) > n_iter else all_candidates
#
#     results = []
#     best_mae = float("inf")
#     best_params = None
#     # Create a copy of validation windows and induce additional missingness.
#     val_windows_missing = val_windows.copy()
#     missing_mask = np.random.rand(*val_windows_missing.shape) < missing_rate
#     val_windows_missing[missing_mask] = np.nan
#
#     for candidate in candidates:
#         try:
#             model = model_class(n_steps=common_params["window_size"],
#                                 n_features=train_windows.shape[-1],
#                                 **candidate,
#                                 batch_size=common_params["batch_size"],
#                                 epochs=TUNING_EPOCHS,  # Use tuning epochs here
#                                 patience=common_params["patience"],
#                                 num_workers=common_params["num_workers"],
#                                 device=common_params["device"],
#                                 saving_path=None,
#                                 model_saving_strategy=None)
#             model.fit({"X": train_windows})
#             mae, _ = evaluate_imputation_windowed(val_windows, val_windows_missing, model, train_std, train_mean)
#             trial = candidate.copy()
#             trial["MAE"] = mae
#             results.append(trial)
#             if mae < best_mae:
#                 best_mae = mae
#                 best_params = candidate
#         except Exception as ex:
#             print(f"Error tuning {model_name} with candidate {candidate}: {ex}")
#     return best_params, results

def simulate_MCAR(data, missing_prob=0.1):
    d = data.copy()
    mask = np.random.rand(*d.shape) < missing_prob
    d[mask] = np.nan
    return d

def simulate_MAR(data, col, threshold_quantile=0.2, missing_prob=0.5):
    d = data.copy()
    threshold = d[col].quantile(threshold_quantile)
    condition = d[col] < threshold
    drop_mask = np.random.rand(condition.sum()) < missing_prob
    indices = d[condition].index[drop_mask]
    d.loc[indices, col] = np.nan
    return d

def simulate_MNAR(data, col, high_quantile=0.8, missing_prob=0.5):
    d = data.copy()
    threshold = d[col].quantile(high_quantile)
    condition = d[col] > threshold
    drop_mask = np.random.rand(condition.sum()) < missing_prob
    indices = d[condition].index[drop_mask]
    d.loc[indices, col] = np.nan
    return d

def simulate_block_missingness(data, col=None, block_length=100, block_start=None):
    d = data.copy()
    if block_start is None:
        max_start = d.shape[0] - block_length
        block_start = np.random.randint(0, max_start)
    block_end = block_start + block_length
    if col is None:
        d.iloc[block_start:block_end] = np.nan
    else:
        idx = d.columns.get_loc(col)
        d.iloc[block_start:block_end, idx] = np.nan
    return d, block_start, block_end
