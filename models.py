# models.py
import os
import pickle
import copyreg
import _thread
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from config import MODEL_DIR, COMMON_PARAMS
from data_utils import create_forecast_pairs

# Import models
from pypots.imputation.usgan.model import USGAN
from pypots.imputation.transformer.model import Transformer as ImputationTransformer
from pypots.imputation.csdi.model import CSDI
from pypots.imputation.saits.model import SAITS
from pypots.imputation.stemgnn.model import StemGNN
from pypots.imputation.brits.model import BRITS
from pypots.imputation.frets.model import FreTS
from pypots.imputation.micn.model import MICN
from pypots.imputation.gpvae.model import GPVAE
from pypots.forecasting.transformer.model import Transformer as ForecastingTransformer

SEED = 42
torch.manual_seed(SEED)

# Make thread locks picklable
copyreg.pickle(_thread.LockType, lambda lock: (_thread.allocate_lock, ()))

# Directory for model checkpoints and TensorBoard logs
LOG_DIR = os.path.join(MODEL_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


# ---------------------------
# Robust save/load utilities
# ---------------------------
def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _build_state_path(pkl_path: str) -> str:
    root, _ = os.path.splitext(pkl_path)
    return root + ".pt"


def _is_valid_pickle(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) > 128
    except Exception:
        return False


def _load_pickle_if_valid(path: str):
    """Return loaded object or None (never raise)."""
    if not _is_valid_pickle(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"⚠️ Failed to unpickle {path}: {e}")
        return None


def _atomic_pickle_dump(obj, path: str):
    """Atomic write to avoid truncated pickles."""
    _ensure_dir(path)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def _safe_pickle_model(model, path: str):
    """Try to pickle a model; if it fails (e.g., thread handles), skip it."""
    try:
        _atomic_pickle_dump(model, path)
        print(f"Saved full model object → {path}")
    except Exception as e:
        print(f"⚠️ {type(model).__name__} not pickleable; skipping model dump ({e}).")


def _save_state_dict_if_any(model, path_pt: str):
    """If model has a .model nn.Module, save its state_dict safely."""
    try:
        nn_module = getattr(model, "model", None)
        if isinstance(nn_module, nn.Module):
            _ensure_dir(path_pt)
            torch.save(nn_module.state_dict(), path_pt)
            print(f"Saved state_dict → {path_pt}")
    except Exception as e:
        print(f"⚠️ Could not save state_dict to {path_pt}: {e}")


def _load_state_dict_if_any(model, path_pt: str) -> bool:
    """Load state_dict if available; returns True on success."""
    try:
        if os.path.exists(path_pt):
            nn_module = getattr(model, "model", None)
            if isinstance(nn_module, nn.Module):
                state = torch.load(path_pt, map_location="cpu")
                nn_module.load_state_dict(state)
                nn_module.eval()
                print(f"Loaded state_dict ← {path_pt}")
                return True
    except Exception as e:
        print(f"⚠️ Failed to load state_dict from {path_pt}: {e}")
    return False


def _try_load_or_none(pkl_path: str, ctor):
    """
    Try loading:
      1) full pickle (.pkl)
      2) state_dict (.pt) into a freshly constructed model (ctor())
    Return model or None.
    """
    # 1) full pickle
    obj = _load_pickle_if_valid(pkl_path)
    if obj is not None:
        print(f"Loaded model from pickle ← {pkl_path}")
        return obj

    # 2) state_dict
    pt_path = _build_state_path(pkl_path)
    try:
        model = ctor()  # fresh instance with correct hyperparams
    except Exception as e:
        print(f"⚠️ Could not construct model for state_dict load: {e}")
        return None

    if _load_state_dict_if_any(model, pt_path):
        return model

    # Nothing to load
    return None


def _save_loss_curve(base_model_path: str, losses):
    """Save a simple loss curve next to the model path."""
    try:
        root, _ = os.path.splitext(base_model_path)
        fig_path = root + "_loss.png"
        plt.figure(figsize=(6, 4))
        plt.plot(list(range(1, len(losses) + 1)), losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"Saved loss curve → {fig_path}")
    except Exception as e:
        print(f"⚠️ Failed to save loss curve: {e}")


# ---------------------------
# Imputation models
# ---------------------------
def train_or_load_usgan(ws, nf, data, model_path):
    logp = os.path.join(LOG_DIR, "USGAN")
    os.makedirs(logp, exist_ok=True)

    def ctor():
        return USGAN(
            n_steps=ws,
            n_features=nf,
            rnn_hidden_size=64,
            lambda_mse=1.0,
            hint_rate=0.9,
            dropout=0.2,
            batch_size=COMMON_PARAMS["batch_size"],
            epochs=COMMON_PARAMS["epochs"],
            patience=COMMON_PARAMS["patience"],
            num_workers=COMMON_PARAMS["num_workers"],
            device=COMMON_PARAMS["device"],
            saving_path=logp,
            model_saving_strategy="best",
        )

    model = _try_load_or_none(model_path, ctor)
    if model is not None:
        return model

    # Train
    model = ctor()
    model.fit({"X": data})
    _safe_pickle_model(model, model_path)
    _save_state_dict_if_any(model, _build_state_path(model_path))
    print("US-GAN training complete.")
    return model


def train_or_load_transformer(ws, nf, data, model_path, tp, td):
    logp = os.path.join(LOG_DIR, "TransformerImpute")
    os.makedirs(logp, exist_ok=True)

    # cast hyperparams
    n_layers = int(tp.get("n_layers", td["n_layers"]))
    d_model = int(tp.get("d_model", td["d_model"]))
    n_heads = int(tp.get("n_heads", td["n_heads"]))
    d_k = int(tp.get("d_k", td["d_k"]))
    d_v = int(tp.get("d_v", td["d_v"]))
    d_ffn = int(tp.get("d_ffn", td["d_ffn"]))
    dropout = float(tp.get("dropout", td["dropout"]))
    attn_dropout = float(tp.get("attn_dropout", td["attn_dropout"]))

    def ctor():
        return ImputationTransformer(
            n_steps=ws,
            n_features=nf,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ffn=d_ffn,
            dropout=dropout,
            attn_dropout=attn_dropout,
            ORT_weight=1.0,
            MIT_weight=1.0,
            batch_size=COMMON_PARAMS["batch_size"],
            epochs=COMMON_PARAMS["epochs"],
            patience=COMMON_PARAMS["patience"],
            num_workers=COMMON_PARAMS["num_workers"],
            device=COMMON_PARAMS["device"],
            saving_path=logp,
            model_saving_strategy="best",
            verbose=False,
        )

    model = _try_load_or_none(model_path, ctor)
    if model is not None:
        return model

    model = ctor()
    model.fit({"X": data})
    _safe_pickle_model(model, model_path)
    _save_state_dict_if_any(model, _build_state_path(model_path))
    print("Transformer training complete.")
    return model


def train_or_load_csdi(ws, nf, data, model_path, tp, td):
    logp = os.path.join(LOG_DIR, "CSDI")
    os.makedirs(logp, exist_ok=True)

    args = {k: int(tp.get(k, td[k])) for k in (
        "n_layers", "n_heads", "n_channels",
        "d_time_embedding", "d_feature_embedding", "d_diffusion_embedding"
    )}
    beta_start = float(tp.get("beta_start", 0.01))
    beta_end = float(tp.get("beta_end", 0.02))

    def ctor():
        return CSDI(
            n_steps=ws,
            n_features=nf,
            **args,
            is_unconditional=False,
            target_strategy="mix",
            n_diffusion_steps=100,
            schedule="linear",
            beta_start=beta_start,
            beta_end=beta_end,
            batch_size=COMMON_PARAMS["batch_size"],
            epochs=COMMON_PARAMS["epochs"],
            patience=COMMON_PARAMS["patience"],
            num_workers=COMMON_PARAMS["num_workers"],
            device=COMMON_PARAMS["device"],
            saving_path=logp,
            model_saving_strategy="best",
            verbose=False,
        )

    model = _try_load_or_none(model_path, ctor)
    if model is not None:
        return model

    model = ctor()
    model.fit({"X": data})
    _safe_pickle_model(model, model_path)
    _save_state_dict_if_any(model, _build_state_path(model_path))
    print("CSDI training complete.")
    return model


def train_or_load_saits(ws, nf, data, model_path, tp, td):
    logp = os.path.join(LOG_DIR, "SAITS")
    os.makedirs(logp, exist_ok=True)

    params = {k: int(tp.get(k, td[k])) for k in (
        "n_layers", "d_model", "n_heads", "d_k", "d_v", "d_ffn"
    )}
    dr = float(tp.get("dropout", td["dropout"]))
    ad = float(tp.get("attn_dropout", td["attn_dropout"]))

    def ctor():
        return SAITS(
            n_steps=ws,
            n_features=nf,
            **params,
            dropout=dr,
            attn_dropout=ad,
            diagonal_attention_mask=True,
            ORT_weight=1.0,
            MIT_weight=1.0,
            batch_size=COMMON_PARAMS["batch_size"],
            epochs=COMMON_PARAMS["epochs"],
            patience=COMMON_PARAMS["patience"],
            num_workers=COMMON_PARAMS["num_workers"],
            device=COMMON_PARAMS["device"],
            saving_path=logp,
            model_saving_strategy="best",
            verbose=False,
        )

    model = _try_load_or_none(model_path, ctor)
    if model is not None:
        return model

    model = ctor()
    model.fit({"X": data})
    _safe_pickle_model(model, model_path)
    _save_state_dict_if_any(model, _build_state_path(model_path))
    print("SAITS training complete.")
    return model


def train_or_load_stemgnn(ws, nf, data, model_path, tp, td):
    logp = os.path.join(LOG_DIR, "StemGNN")
    os.makedirs(logp, exist_ok=True)

    args = {
        "n_layers": int(tp.get("n_layers", td["n_layers"])),
        "d_model": int(tp.get("d_model", td["d_model"])),
        "dropout": float(tp.get("dropout", td["dropout"])),
        "n_stacks": int(tp.get("n_stacks", td["n_stacks"])),
    }

    def ctor():
        return StemGNN(
            n_steps=ws,
            n_features=nf,
            **args,
            batch_size=COMMON_PARAMS["batch_size"],
            epochs=COMMON_PARAMS["epochs"],
            patience=COMMON_PARAMS["patience"],
            num_workers=COMMON_PARAMS["num_workers"],
            device=COMMON_PARAMS["device"],
            saving_path=logp,
            model_saving_strategy="best",
            verbose=False,
        )

    model = _try_load_or_none(model_path, ctor)
    if model is not None:
        return model

    model = ctor()
    model.fit({"X": data})
    _safe_pickle_model(model, model_path)
    _save_state_dict_if_any(model, _build_state_path(model_path))
    print("StemGNN training complete.")
    return model


def train_or_load_brits(ws, nf, data, model_path, tp, td):
    logp = os.path.join(LOG_DIR, "BRITS")
    os.makedirs(logp, exist_ok=True)

    rnn_sz = int(tp.get("rnn_hidden_size", td["rnn_hidden_size"]))

    def ctor():
        return BRITS(
            n_steps=ws,
            n_features=nf,
            rnn_hidden_size=rnn_sz,
            batch_size=COMMON_PARAMS["batch_size"],
            epochs=COMMON_PARAMS["epochs"],
            patience=COMMON_PARAMS["patience"],
            num_workers=COMMON_PARAMS["num_workers"],
            device=COMMON_PARAMS["device"],
            saving_path=logp,
            model_saving_strategy="best",
            verbose=False,
        )

    model = _try_load_or_none(model_path, ctor)
    if model is not None:
        return model

    model = ctor()
    model.fit({"X": data})
    _safe_pickle_model(model, model_path)
    _save_state_dict_if_any(model, _build_state_path(model_path))
    print("BRITS training complete.")
    return model


def train_or_load_frets(ws, nf, data, model_path, tp, td):
    logp = os.path.join(LOG_DIR, "FreTS")
    os.makedirs(logp, exist_ok=True)

    es = int(tp.get("embed_size", td["embed_size"]))
    hs = int(tp.get("hidden_size", td["hidden_size"]))
    ci = bool(tp.get("channel_independence", td["channel_independence"]))

    def ctor():
        return FreTS(
            n_steps=ws,
            n_features=nf,
            embed_size=es,
            hidden_size=hs,
            channel_independence=ci,
            ORT_weight=1.0,
            MIT_weight=1.0,
            batch_size=COMMON_PARAMS["batch_size"],
            epochs=COMMON_PARAMS["epochs"],
            patience=COMMON_PARAMS["patience"],
            num_workers=COMMON_PARAMS["num_workers"],
            device=COMMON_PARAMS["device"],
            saving_path=logp,
            model_saving_strategy="best",
            verbose=False,
        )

    model = _try_load_or_none(model_path, ctor)
    if model is not None:
        return model

    model = ctor()
    model.fit({"X": data})
    _safe_pickle_model(model, model_path)
    _save_state_dict_if_any(model, _build_state_path(model_path))
    print("FreTS training complete.")
    return model


def train_or_load_micn(ws, nf, data, state_dict_path, tp, td):
    """MICN historically saved only state_dict; keep that pattern (plus a .pkl if possible)."""
    logp = os.path.join(LOG_DIR, "MICN")
    os.makedirs(logp, exist_ok=True)

    n_layers = int(tp.get("n_layers", td["n_layers"]))
    d_model = int(tp.get("d_model", td["d_model"]))
    raw_kernel = tp.get("conv_kernel", td["conv_kernel"])
    if not isinstance(raw_kernel, (list, tuple)):
        raw_kernel = (raw_kernel,)
    conv_kernel = [int(k) for k in raw_kernel]

    def ctor():
        return MICN(
            n_steps=ws,
            n_features=nf,
            n_layers=n_layers,
            d_model=d_model,
            conv_kernel=conv_kernel,
            dropout=0.0,
            ORT_weight=1.0,
            MIT_weight=1.0,
            batch_size=COMMON_PARAMS["batch_size"],
            epochs=COMMON_PARAMS["epochs"],
            patience=COMMON_PARAMS["patience"],
            num_workers=COMMON_PARAMS["num_workers"],
            device=COMMON_PARAMS["device"],
            saving_path=logp,
            model_saving_strategy="best",
            verbose=False,
        )

    # Try loading state_dict or pkl next to it (.pkl with same root)
    pkl_path = state_dict_path + ".pkl"
    model = _try_load_or_none(pkl_path, ctor)
    if model is not None:
        return model

    # Try old state_dict path explicitly (backward-compatible)
    if os.path.exists(state_dict_path):
        model = ctor()
        try:
            state = torch.load(state_dict_path, map_location="cpu")
            model.model.load_state_dict(state)
            model.model.eval()
            print(f"Loaded MICN weights from disk ← {state_dict_path}")
            _safe_pickle_model(model, pkl_path)
            return model
        except Exception as e:
            print(f"⚠️ Failed loading MICN state_dict from {state_dict_path}: {e}")

    # Train
    model = ctor()
    model.fit({"X": data})
    try:
        torch.save(model.model.state_dict(), state_dict_path)
        print(f"MICN weights saved → {state_dict_path}")
    except Exception as e:
        print(f"⚠️ Could not save MICN state_dict: {e}")
    _safe_pickle_model(model, pkl_path)
    print("MICN training complete.")
    return model


def train_or_load_gpvae(window_size, n_features, train_windows,
                        gpvae_model_path, tuned_params, tuning_defaults):
    log_dir = os.path.join(LOG_DIR, "GPVAE")
    os.makedirs(log_dir, exist_ok=True)

    # Hyperparams (kernel_scales is an int)
    latent_size = int(tuned_params.get("latent_size", tuning_defaults.get("latent_size", 16)))
    raw_enc = tuned_params.get("encoder_sizes", tuning_defaults.get("encoder_sizes", (64, 32)))
    encoder_sizes = tuple(int(x) for x in (raw_enc if isinstance(raw_enc, (list, tuple)) else [raw_enc]))
    raw_dec = tuned_params.get("decoder_sizes", tuning_defaults.get("decoder_sizes", (32, 64)))
    decoder_sizes = tuple(int(x) for x in (raw_dec if isinstance(raw_dec, (list, tuple)) else [raw_dec]))
    beta = float(tuned_params.get("beta", tuning_defaults.get("beta", 1.0)))
    M = int(tuned_params.get("M", tuning_defaults.get("M", 1)))
    K = int(tuned_params.get("K", tuning_defaults.get("K", 1)))
    kernel = str(tuned_params.get("kernel", tuning_defaults.get("kernel", "rbf")))
    sigma = float(tuned_params.get("sigma", tuning_defaults.get("sigma", 1.0)))
    length_scale = float(tuned_params.get("length_scale", tuning_defaults.get("length_scale", 1.0)))
    kernel_scales = int(tuned_params.get("kernel_scales", tuning_defaults.get("kernel_scales", 1)))

    def ctor():
        return GPVAE(
            n_steps=window_size,
            n_features=n_features,
            latent_size=latent_size,
            encoder_sizes=encoder_sizes,
            decoder_sizes=decoder_sizes,
            beta=beta,
            M=M,
            K=K,
            kernel=kernel,
            sigma=sigma,
            length_scale=length_scale,
            kernel_scales=kernel_scales,
            window_size=window_size,
            batch_size=COMMON_PARAMS["batch_size"],
            epochs=COMMON_PARAMS["epochs"],
            patience=COMMON_PARAMS["patience"],
            num_workers=COMMON_PARAMS["num_workers"],
            device=COMMON_PARAMS["device"],
            saving_path=log_dir,
            model_saving_strategy="best",
            verbose=False,
        )

    model = _try_load_or_none(gpvae_model_path, ctor)
    if model is not None:
        return model

    # Train + save
    model = ctor()
    history = model.fit({"X": train_windows})
    # Try to save loss curve if Keras-like history is present
    losses = getattr(getattr(history, "history", {}), "get", lambda *_: [])("loss", [])
    if not losses and isinstance(history, dict):
        losses = history.get("loss", [])
    if losses:
        _save_loss_curve(gpvae_model_path, losses)

    _safe_pickle_model(model, gpvae_model_path)
    _save_state_dict_if_any(model, _build_state_path(gpvae_model_path))
    print("GPVAE training complete.")
    return model


def load_or_train_forecasting_transformer(window_size, n_features, train_df, forecast_transformer_model_path):
    def ctor():
        return ForecastingTransformer(
            n_steps=window_size,
            n_features=n_features,
            n_pred_steps=1,
            n_pred_features=n_features,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_model=64,
            n_heads=4,
            d_k=16,
            d_v=16,
            d_ffn=128,
            dropout=0.1,
            attn_dropout=0.1,
            batch_size=32,
            epochs=COMMON_PARAMS["epochs"],
            patience=None,
            num_workers=0,
            device=COMMON_PARAMS["device"],
            saving_path=None,
            model_saving_strategy=None,
            verbose=False,
        )

    model = _try_load_or_none(forecast_transformer_model_path, ctor)
    if model is not None:
        print("Loaded Forecasting Transformer model from disk.")
        return model

    model = ctor()
    X_train_ft, y_train_ft = create_forecast_pairs(train_df.values, window_size, horizon=1)
    history = model.fit({"X": X_train_ft, "y": y_train_ft, "X_pred": X_train_ft})
    # Save loss curve if available
    losses = getattr(getattr(history, 'history', {}), 'get', lambda *_: [])('loss', [])
    if losses:
        _save_loss_curve(forecast_transformer_model_path, losses)

    _safe_pickle_model(model, forecast_transformer_model_path)
    _save_state_dict_if_any(model, _build_state_path(forecast_transformer_model_path))
    print("Forecasting Transformer training complete and model saved.")
    return model
