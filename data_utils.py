# data_utils.py
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_FILE


def load_and_prepare_data(selected_cols=None, split_ratios=(0.6, 0.2, 0.2)):
    """
    Load the CSV data, select columns, split into train/val/test, and normalize.
    """
    df_full = pd.read_csv(DATA_FILE, parse_dates=["TimeStamp"])
    df_full.set_index("TimeStamp", inplace=True)
    if selected_cols is None:
        selected_cols = list(df_full.columns)[:6]
    df_full = df_full[selected_cols]
    print("Full data loaded. Shape:", df_full.shape)
    n_total = len(df_full)
    train_df_orig = df_full.iloc[:int(split_ratios[0] * n_total)]
    val_df_orig = df_full.iloc[int(split_ratios[0] * n_total):int((split_ratios[0] + split_ratios[1]) * n_total)]
    test_df_orig = df_full.iloc[int((split_ratios[0] + split_ratios[1]) * n_total):]
    print("Train shape:", train_df_orig.shape, "Validation shape:", val_df_orig.shape, "Test shape:",
          test_df_orig.shape)

    # For forecasting, drop NaNs in training set.
    train_df_orig = train_df_orig.dropna()
    train_mean = train_df_orig.mean()
    train_std = train_df_orig.std()

    train_df = (train_df_orig - train_mean) / train_std
    val_df = (val_df_orig - train_mean) / train_std
    test_df = (test_df_orig - train_mean) / train_std

    return train_df_orig, train_df, val_df, test_df, test_df_orig, train_mean, train_std, selected_cols


def create_windows(df, window_size):
    """Create sliding windows from a DataFrame."""
    data = df.values
    n_windows = data.shape[0] - window_size + 1
    return np.array([data[i:i + window_size] for i in range(n_windows)])


def process_prediction_result(result):
    """
    Process prediction outputs.
    If result is a dict, extract the "X" key (or first available value).
    """
    if isinstance(result, dict):
        out = np.array(result["X"]) if "X" in result else np.array(next(iter(result.values())))
    else:
        out = np.array(result)
    if out.ndim == 4:
        out = np.mean(out, axis=1)
    if out.ndim == 2:
        out = out[:, np.newaxis, :]
    return out


def reconstruct_series(windowed_preds, window_size, original_length):
    """
    Reconstruct a full series from overlapping window predictions by averaging.
    Expects windowed_preds to have shape (n_windows, window_size, n_features).
    """
    n_features = windowed_preds.shape[-1]
    sum_series = np.zeros((original_length, n_features))
    count_series = np.zeros(original_length)
    n_windows = windowed_preds.shape[0]
    for i in range(n_windows):
        sum_series[i:i + window_size] += windowed_preds[i]
        count_series[i:i + window_size] += 1
    return sum_series / (count_series[:, None] + 1e-8)


def make_plain_label(s) -> str:
    """
    Convert labels that may contain LaTeX (e.g., '$\\text{Lobby PM}_{10}$')
    into plain text so Matplotlib's mathtext won't crash.
    """
    s = str(s)
    # remove math delimiters
    s = s.replace("$", "")
    # \text{...} -> ...
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    # generic command removal like \mathrm{...} -> ...
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
    # _{10} -> _10
    s = re.sub(r"_\{([^}]*)\}", r"_\1", s)
    # unescape \% and remove stray braces
    s = s.replace(r"\%", "%").replace("{", "").replace("}", "")
    return s


def generate_heatmap(df, title, save_path):
    """Generate and save a heatmap from a DataFrame with safe (non-LaTeX) labels."""
    # Sanitize labels to avoid mathtext parsing errors
    df_plot = df.copy()
    df_plot.columns = [make_plain_label(c) for c in df_plot.columns]
    df_plot.index = [make_plain_label(i) for i in df_plot.index]

    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1.5, style="whitegrid", rc={"font.family": "Times New Roman"})
    hmap = sns.heatmap(df_plot, annot=True, cmap="viridis", fmt=".1f")
    for label in hmap.get_xticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(18)
        label.set_rotation(45)
    for label in hmap.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(18)
        label.set_rotation(0)
    hmap.set_xlabel("Variable", fontsize=18, family="Times New Roman")
    hmap.set_ylabel("Scenario", fontsize=18, family="Times New Roman")
    hmap.set_title(title, fontsize=22, family="Times New Roman")
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()


def create_forecast_pairs(series, window_size, horizon=1):
    """
    Generate forecasting (X, y) pairs from a 1D numpy array.
    """
    X, y = [], []
    n = len(series)
    for i in range(n - window_size - horizon + 1):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size + horizon - 1])
    return np.array(X), np.array(y)
