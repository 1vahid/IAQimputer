# config.py
import os
import matplotlib.pyplot as plt

# Global font and matplotlib settings.
plt.rc("font", family="Times New Roman")
plt.rcParams["mathtext.fontset"] = "stix"  # Render math text with a Times-like style.

# Data paths.
DATA_DIR = "analysis_results"
DATA_FILE = os.path.join(DATA_DIR, "Concatenated_Indoor_Data_5min.csv")

# Model saving directory.
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Forecasting Transformer model path.
FORECAST_TRANSFORMER_MODEL_PATH = os.path.join(MODEL_DIR, "forecast_transformer_model.pkl")

# Advanced tuning results directory.
OPT_DIR = "optimization_results"
os.makedirs(OPT_DIR, exist_ok=True)

# Separate epoch parameters.
TRAINING_EPOCHS = 60       # Full training epochs.
TUNING_EPOCHS = 5         # Fewer epochs during hyperparameter tuning.

# Common training parameters for final training.
COMMON_PARAMS = {
    "window_size": 10,
    "batch_size": 32,
    "epochs": TRAINING_EPOCHS,  # Use full training epochs here.
    "patience": None,
    "num_workers": 0,
    "device": "cpu"
}

# Rename map for plotting.
RENAME_MAP = {
    "Lobby_PM10_Indoor": r"$\text{Lobby PM}_{10}$",
    "Lobby_PM2_5_Indoor": r"$\text{Lobby PM}_{2.5}$",
    "Lobby_PM1_Indoor": r"$\text{Lobby PM}_{1}$",
    "Platform_PM10_Indoor": r"$\text{Platform PM}_{10}$",
    "Platform_PM2_5_Indoor": r"$\text{Platform PM}_{2.5}$",
    "Platform_PM1_Indoor": r"$\text{Platform PM}_{1}$"
}
