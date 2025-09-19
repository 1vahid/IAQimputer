import os
import pickle
import copyreg
import _thread
import torch

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

def _safe_pickle_model(model, path):
    """Try to pickle a model; if it fails (e.g. thread handles), skip it."""
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f)
    except TypeError:
        print(f"⚠️ {type(model).__name__} not pickleable; skipping model dump.")

# --- Imputation models ---

def train_or_load_usgan(ws, nf, data, model_path):
    name, logp = "USGAN", os.path.join(LOG_DIR, "USGAN")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("Loaded US-GAN model from disk.")
        return model

    os.makedirs(logp, exist_ok=True)
    model = USGAN(
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
        model_saving_strategy="best"
    )
    model.fit({"X": data})
    _safe_pickle_model(model, model_path)
    print("US-GAN training complete.")
    return model

def train_or_load_transformer(ws, nf, data, model_path, tp, td):
    name, logp = "TransformerImpute", os.path.join(LOG_DIR, "TransformerImpute")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("Loaded Transformer model from disk.")
        return model

    # cast hyperparams
    n_layers     = int(tp.get("n_layers",     td["n_layers"]))
    d_model      = int(tp.get("d_model",      td["d_model"]))
    n_heads      = int(tp.get("n_heads",      td["n_heads"]))
    d_k          = int(tp.get("d_k",          td["d_k"]))
    d_v          = int(tp.get("d_v",          td["d_v"]))
    d_ffn        = int(tp.get("d_ffn",        td["d_ffn"]))
    dropout      = float(tp.get("dropout",     td["dropout"]))
    attn_dropout = float(tp.get("attn_dropout",td["attn_dropout"]))

    os.makedirs(logp, exist_ok=True)
    model = ImputationTransformer(
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
        verbose=False
    )
    model.fit({"X": data})
    _safe_pickle_model(model, model_path)
    print("Transformer training complete.")
    return model

def train_or_load_csdi(ws, nf, data, model_path, tp, td):
    name, logp = "CSDI", os.path.join(LOG_DIR, "CSDI")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("Loaded CSDI model from disk.")
        return model

    # cast hyperparams
    args = {k: int(tp.get(k, td[k])) for k in (
        "n_layers","n_heads","n_channels",
        "d_time_embedding","d_feature_embedding","d_diffusion_embedding"
    )}
    beta_start = float(tp.get("beta_start", 0.01))
    beta_end   = float(tp.get("beta_end",   0.02))

    os.makedirs(logp, exist_ok=True)
    model = CSDI(
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
        verbose=False
    )
    model.fit({"X": data})
    _safe_pickle_model(model, model_path)
    print("CSDI training complete.")
    return model

def train_or_load_saits(ws, nf, data, model_path, tp, td):
    name, logp = "SAITS", os.path.join(LOG_DIR, "SAITS")
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print("Loaded SAITS model from disk.")
            return model
        except:
            pass

    params = {k: int(tp.get(k, td[k])) for k in (
        "n_layers","d_model","n_heads","d_k","d_v","d_ffn"
    )}
    dr = float(tp.get("dropout", td["dropout"]))
    ad = float(tp.get("attn_dropout", td["attn_dropout"]))

    os.makedirs(logp, exist_ok=True)
    model = SAITS(
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
        verbose=False
    )
    model.fit({"X": data})
    _safe_pickle_model(model, model_path)
    print("SAITS training complete.")
    return model

def train_or_load_stemgnn(ws, nf, data, model_path, tp, td):
    name, logp = "StemGNN", os.path.join(LOG_DIR, "StemGNN")
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print("Loaded StemGNN model from disk.")
            return model
        except:
            pass

    args = {
        "n_layers": int(tp.get("n_layers", td["n_layers"])),
        "d_model":  int(tp.get("d_model", td["d_model"])),
        "dropout":  float(tp.get("dropout", td["dropout"])),
        "n_stacks": int(tp.get("n_stacks", td["n_stacks"]))
    }

    os.makedirs(logp, exist_ok=True)
    model = StemGNN(
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
        verbose=False
    )
    model.fit({"X": data})
    _safe_pickle_model(model, model_path)
    print("StemGNN training complete.")
    return model

def train_or_load_brits(ws, nf, data, model_path, tp, td):
    name, logp = "BRITS", os.path.join(LOG_DIR, "BRITS")
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print("Loaded BRITS model from disk.")
            return model
        except:
            pass

    rnn_sz = int(tp.get("rnn_hidden_size", td["rnn_hidden_size"]))

    os.makedirs(logp, exist_ok=True)
    model = BRITS(
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
        verbose=False
    )
    model.fit({"X": data})
    _safe_pickle_model(model, model_path)
    print("BRITS training complete.")
    return model

def train_or_load_frets(ws, nf, data, model_path, tp, td):
    name, logp = "FreTS", os.path.join(LOG_DIR, "FreTS")
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print("Loaded FreTS model from disk.")
            return model
        except:
            pass

    es = int(tp.get("embed_size", td["embed_size"]))
    hs = int(tp.get("hidden_size", td["hidden_size"]))
    ci = bool(tp.get("channel_independence", td["channel_independence"]))

    os.makedirs(logp, exist_ok=True)
    model = FreTS(
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
        verbose=False
    )
    model.fit({"X": data})
    _safe_pickle_model(model, model_path)
    print("FreTS training complete.")
    return model

def train_or_load_micn(ws, nf, data, state_dict_path, tp, td):
    name, logp = "MICN", os.path.join(LOG_DIR, "MICN")

    # cast hyperparams
    n_layers = int(tp.get("n_layers", td["n_layers"]))
    d_model  = int(tp.get("d_model",  td["d_model"]))
    raw_kernel = tp.get("conv_kernel", td["conv_kernel"])
    if not isinstance(raw_kernel, (list, tuple)):
        raw_kernel = (raw_kernel,)
    conv_kernel = [int(k) for k in raw_kernel]

    os.makedirs(logp, exist_ok=True)
    model = MICN(
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
        verbose=False
    )

    if os.path.exists(state_dict_path):
        state = torch.load(state_dict_path, map_location="cpu")
        model.model.load_state_dict(state)
        model.model.eval()
        print("Loaded MICN weights from disk.")
        _safe_pickle_model(model, state_dict_path + ".pkl")
        return model

    model.fit({"X": data})
    torch.save(model.model.state_dict(), state_dict_path)
    _safe_pickle_model(model, state_dict_path)
    print("MICN training complete and weights saved.")
    return model

def train_or_load_gpvae(window_size, n_features, train_windows,
                        gpvae_model_path, tuned_params, tuning_defaults):
    # 1) Prepare log directory for GPVAE
    log_dir = os.path.join(LOG_DIR, "GPVAE")
    os.makedirs(log_dir, exist_ok=True)

    # 2) Attempt to load existing model
    if os.path.exists(gpvae_model_path):
        try:
            with open(gpvae_model_path, "rb") as f:
                model = pickle.load(f)
            print("Loaded GPVAE model from disk.")
            return model
        except Exception as e:
            print("Failed to load GPVAE model:", e)

    # 3) Cast hyperparameters (kernel_scales must be an int)
    latent_size  = int(tuned_params.get("latent_size",   tuning_defaults.get("latent_size", 16)))
    raw_enc      = tuned_params.get("encoder_sizes",   tuning_defaults.get("encoder_sizes", (64,32)))
    encoder_sizes= tuple(int(x) for x in (raw_enc if isinstance(raw_enc,(list,tuple)) else [raw_enc]))
    raw_dec      = tuned_params.get("decoder_sizes",   tuning_defaults.get("decoder_sizes", (32,64)))
    decoder_sizes= tuple(int(x) for x in (raw_dec if isinstance(raw_dec,(list,tuple)) else [raw_dec]))
    beta         = float(tuned_params.get("beta",        tuning_defaults.get("beta",1.0)))
    M             = int(tuned_params.get("M",           tuning_defaults.get("M",1)))
    K             = int(tuned_params.get("K",           tuning_defaults.get("K",1)))
    kernel       = str(tuned_params.get("kernel",
                        tuning_defaults.get("kernel","rbf")))
    sigma        = float(tuned_params.get("sigma",       tuning_defaults.get("sigma",1.0)))
    length_scale = float(tuned_params.get("length_scale",tuning_defaults.get("length_scale",1.0)))
    # enforce integer, not a list
    kernel_scales= int(tuned_params.get("kernel_scales",
                             tuning_defaults.get("kernel_scales",1)))

    # 4) Instantiate GPVAE with saving_path for TensorBoard logs
    model = GPVAE(
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
        kernel_scales=kernel_scales,   # now an int
        window_size=window_size,
        batch_size=COMMON_PARAMS["batch_size"],
        epochs=COMMON_PARAMS["epochs"],
        patience=COMMON_PARAMS["patience"],
        num_workers=COMMON_PARAMS["num_workers"],
        device=COMMON_PARAMS["device"],
        saving_path=log_dir,           # TensorBoard logs + checkpoints
        model_saving_strategy="best",
        verbose=False
    )

    # 5) Train (TensorBoard will capture loss curves)
    model.fit({"X": train_windows})

    # 6) Safely pickle the model wrapper (ignoring thread-handle errors)
    try:
        with open(gpvae_model_path, "wb") as f:
            pickle.dump(model, f)
    except TypeError:
        print("⚠️ GPVAE model contains unpickleable internals; skipping full-object dump.")

    print("GPVAE training complete.")
    return model


def load_or_train_forecasting_transformer(window_size, n_features, train_df, forecast_transformer_model_path):
    if os.path.exists(forecast_transformer_model_path):
        with open(forecast_transformer_model_path, "rb") as f:
            model = pickle.load(f)
        print("Loaded Forecasting Transformer model from disk.")
    else:
        model = ForecastingTransformer(
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
            device="cpu",
            saving_path=None,
            model_saving_strategy=None,
            verbose=False
        )
        X_train_ft, y_train_ft = create_forecast_pairs(train_df.values, window_size, horizon=1)
        history = model.fit({"X": X_train_ft, "y": y_train_ft, "X_pred": X_train_ft})
        losses = getattr(history, 'history', {}).get('loss', [])
        if losses:
            _save_loss_curve(forecast_transformer_model_path, losses)
        with open(forecast_transformer_model_path, "wb") as f:
            pickle.dump(model, f)
        print("Forecasting Transformer training complete and model saved.")
    return model
