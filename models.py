# models.py
import os
import pickle
from config import MODEL_DIR, COMMON_PARAMS


# Import deep imputation and forecasting models.
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

def train_or_load_usgan(window_size, n_features, train_windows, usgan_model_path):
    if os.path.exists(usgan_model_path):
        with open(usgan_model_path, "rb") as f:
            model = pickle.load(f)
        print("Loaded US-GAN model from disk.")
    else:
        model = USGAN(n_steps=window_size,
                      n_features=n_features,
                      rnn_hidden_size=64,
                      lambda_mse=1.0,
                      hint_rate=0.9,
                      dropout=0.2,
                      batch_size=COMMON_PARAMS["batch_size"],
                      epochs=COMMON_PARAMS["epochs"],
                      patience=COMMON_PARAMS["patience"],
                      num_workers=COMMON_PARAMS["num_workers"],
                      device=COMMON_PARAMS["device"],
                      saving_path=None,
                      model_saving_strategy=None)
        model.fit({"X": train_windows})
        with open(usgan_model_path, "wb") as f:
            pickle.dump(model, f)
        print("US-GAN training complete and model saved.")
    return model

def train_or_load_transformer(window_size, n_features, train_windows, transformer_model_path, tuned_params, tuning_defaults):
    if os.path.exists(transformer_model_path):
        with open(transformer_model_path, "rb") as f:
            model = pickle.load(f)
        print("Loaded Transformer imputation model from disk.")
    else:
        model = ImputationTransformer(n_steps=window_size,
                                      n_features=n_features,
                                      n_layers=tuned_params["n_layers"],
                                      d_model=tuned_params["d_model"],
                                      n_heads=tuned_params["n_heads"],
                                      d_k=tuned_params["d_k"],
                                      d_v=tuned_params["d_v"],
                                      d_ffn=tuned_params["d_ffn"],
                                      dropout=tuned_params["dropout"],
                                      attn_dropout=tuned_params["attn_dropout"],
                                      ORT_weight=1.0,
                                      MIT_weight=1.0,
                                      batch_size=COMMON_PARAMS["batch_size"],
                                      epochs=COMMON_PARAMS["epochs"],
                                      patience=COMMON_PARAMS["patience"],
                                      num_workers=COMMON_PARAMS["num_workers"],
                                      device=COMMON_PARAMS["device"],
                                      saving_path=None,
                                      model_saving_strategy=None,
                                      verbose=False)
        model.fit({"X": train_windows})
        with open(transformer_model_path, "wb") as f:
            pickle.dump(model, f)
        print("Transformer imputation training complete and model saved.")
    return model

def train_or_load_csdi(window_size, n_features, train_windows, csdi_model_path, tuned_params, tuning_defaults):
    if os.path.exists(csdi_model_path):
        with open(csdi_model_path, "rb") as f:
            model = pickle.load(f)
        print("Loaded CSDI model from disk.")
    else:
        model = CSDI(n_steps=window_size,
                     n_features=n_features,
                     n_layers=tuned_params["n_layers"],
                     n_heads=tuned_params["n_heads"],
                     n_channels=tuned_params["n_channels"],
                     d_time_embedding=tuned_params["d_time_embedding"],
                     d_feature_embedding=tuned_params["d_feature_embedding"],
                     d_diffusion_embedding=tuned_params["d_diffusion_embedding"],
                     is_unconditional=False,
                     target_strategy="mix",
                     n_diffusion_steps=100,
                     schedule="linear",
                     beta_start=0.01,
                     beta_end=0.02,
                     batch_size=COMMON_PARAMS["batch_size"],
                     epochs=COMMON_PARAMS["epochs"],
                     patience=COMMON_PARAMS["patience"],
                     num_workers=COMMON_PARAMS["num_workers"],
                     device=COMMON_PARAMS["device"],
                     saving_path=None,
                     model_saving_strategy=None,
                     verbose=False)
        model.fit({"X": train_windows})
        with open(csdi_model_path, "wb") as f:
            pickle.dump(model, f)
        print("CSDI training complete and model saved.")
    return model

def train_or_load_saits(window_size, n_features, train_windows, saits_model_path, tuned_params, tuning_defaults):
    from pypots.imputation.saits.model import SAITS
    if os.path.exists(saits_model_path):
        try:
            with open(saits_model_path, "rb") as f:
                model = pickle.load(f)
            print("Loaded SAITS model from disk.")
        except Exception as e:
            print("Error loading SAITS model:", e)
            model = None
    else:
        model = SAITS(n_steps=window_size,
                      n_features=n_features,
                      n_layers=tuned_params["n_layers"],
                      d_model=tuned_params["d_model"],
                      n_heads=tuned_params["n_heads"],
                      d_k=tuned_params.get("d_k", tuning_defaults.get("d_k", 64)),
                      d_v=tuned_params.get("d_v", tuning_defaults.get("d_v", 64)),
                      d_ffn=tuned_params["d_ffn"],
                      dropout=tuned_params["dropout"],
                      attn_dropout=tuned_params["attn_dropout"],
                      diagonal_attention_mask=True,
                      ORT_weight=1.0,
                      MIT_weight=1.0,
                      batch_size=COMMON_PARAMS["batch_size"],
                      epochs=COMMON_PARAMS["epochs"],
                      patience=COMMON_PARAMS["patience"],
                      num_workers=COMMON_PARAMS["num_workers"],
                      device=COMMON_PARAMS["device"],
                      saving_path=None,
                      model_saving_strategy=None,
                      verbose=False)
        model.fit({"X": train_windows})
        with open(saits_model_path, "wb") as f:
            pickle.dump(model, f)
        print("SAITS training complete and model saved.")
    return model

def train_or_load_stemgnn(window_size, n_features, train_windows, stemgnn_model_path, tuned_params, tuning_defaults):
    from pypots.imputation.stemgnn.model import StemGNN
    if os.path.exists(stemgnn_model_path):
        try:
            with open(stemgnn_model_path, "rb") as f:
                model = pickle.load(f)
            print("Loaded StemGNN model from disk.")
        except Exception as e:
            print("Error loading StemGNN model:", e)
            model = None
    else:
        n_stacks = tuned_params.get("n_stacks", tuning_defaults.get("n_stacks", 1))
        model = StemGNN(n_steps=window_size,
                        n_features=n_features,
                        n_layers=tuned_params.get("n_layers", tuning_defaults.get("n_layers", 2)),
                        d_model=tuned_params.get("d_model", tuning_defaults.get("d_model", 64)),
                        dropout=tuned_params.get("dropout", tuning_defaults.get("dropout", 0.1)),
                        n_stacks=n_stacks,  # Provide the required argument
                        batch_size=COMMON_PARAMS["batch_size"],
                        epochs=COMMON_PARAMS["epochs"],
                        patience=COMMON_PARAMS["patience"],
                        num_workers=COMMON_PARAMS["num_workers"],
                        device=COMMON_PARAMS["device"],
                        saving_path=None,
                        model_saving_strategy=None,
                        verbose=False)
        model.fit({"X": train_windows})
        with open(stemgnn_model_path, "wb") as f:
            pickle.dump(model, f)
        print("StemGNN training complete and model saved.")
    return model

def train_or_load_brits(window_size, n_features, train_windows, brits_model_path):
    from pypots.imputation.brits.model import BRITS
    if os.path.exists(brits_model_path):
        try:
            with open(brits_model_path, "rb") as f:
                model = pickle.load(f)
            print("Loaded BRITS model from disk.")
        except Exception as e:
            print("Error loading BRITS model:", e)
            model = None
    else:
        try:
            model = BRITS(n_steps=window_size,
                          n_features=n_features,
                          rnn_hidden_size=64,
                          batch_size=COMMON_PARAMS["batch_size"],
                          epochs=COMMON_PARAMS["epochs"],
                          patience=COMMON_PARAMS["patience"],
                          num_workers=COMMON_PARAMS["num_workers"],
                          device=COMMON_PARAMS["device"],
                          saving_path=None,
                          model_saving_strategy=None,
                          verbose=False)
            model.fit({"X": train_windows})
            with open(brits_model_path, "wb") as f:
                pickle.dump(model, f)
            print("BRITS training complete and model saved.")
        except Exception as e:
            print("BRITS training failed:", e)
            model = None
    return model

def train_or_load_frets(window_size, n_features, train_windows, frets_model_path, tuned_params, tuning_defaults):
    from pypots.imputation.frets.model import FreTS
    if os.path.exists(frets_model_path):
        try:
            with open(frets_model_path, "rb") as f:
                model = pickle.load(f)
            print("Loaded FreTS model from disk.")
        except Exception as e:
            print("Error loading FreTS model:", e)
            model = None
    else:
        model = FreTS(n_steps=window_size,
                      n_features=n_features,
                      embed_size=tuned_params.get("embed_size", tuning_defaults.get("embed_size", 128)),
                      hidden_size=tuned_params.get("hidden_size", tuning_defaults.get("hidden_size", 256)),
                      channel_independence=tuned_params.get("channel_independence", tuning_defaults.get("channel_independence", False)),
                      ORT_weight=1.0,
                      MIT_weight=1.0,
                      batch_size=COMMON_PARAMS["batch_size"],
                      epochs=COMMON_PARAMS["epochs"],
                      patience=COMMON_PARAMS["patience"],
                      num_workers=COMMON_PARAMS["num_workers"],
                      device=COMMON_PARAMS["device"],
                      saving_path=None,
                      model_saving_strategy=None,
                      verbose=False)
        model.fit({"X": train_windows})
        with open(frets_model_path, "wb") as f:
            pickle.dump(model, f)
        print("FreTS training complete and model saved.")
    return model


def train_or_load_micn (window_size, n_features, train_windows, micn_model_path, tuned_params, tuning_defaults):
    from pypots.imputation.micn.model import MICN
    if os.path.exists(micn_model_path):
        try:
            with open(micn_model_path, "rb") as f:
                model = pickle.load(f)
            print("Loaded MICN model from disk.")
        except Exception as e:
            print("Error loading MICN model:", e)
            model = None
    else:
        model = MICN(n_steps=window_size,
                     n_features=n_features,
                     n_layers=tuned_params.get("n_layers", tuning_defaults.get("n_layers", 2)),
                     d_model=tuned_params.get("d_model", tuning_defaults.get("d_model", 64)),
                     conv_kernel= tuned_params.get("conv_kernel", tuning_defaults.get("conv_kernel", [3])),
                     dropout=tuned_params.get("dropout", tuning_defaults.get("dropout", 0.1)),
                     ORT_weight=1.0,
                     MIT_weight=1.0,
                     batch_size=COMMON_PARAMS["batch_size"],
                     epochs=COMMON_PARAMS["epochs"],
                     patience=COMMON_PARAMS["patience"],
                     num_workers=COMMON_PARAMS["num_workers"],
                     device=COMMON_PARAMS["device"],
                     saving_path=None,
                     model_saving_strategy=None,
                     verbose=False)
        model.fit({"X": train_windows})
        with open(micn_model_path, "wb") as f:
            pickle.dump(model, f)
        print("MICN training complete and model saved.")
    return model

def train_or_load_gpvae(window_size, n_features, train_windows, gpvae_model_path, tuned_params, tuning_defaults):
    from pypots.imputation.gpvae.model import GPVAE
    if os.path.exists(gpvae_model_path):
        try:
            with open(gpvae_model_path, "rb") as f:
                model = pickle.load(f)
            print("Loaded GPVAE model from disk.")
        except Exception as e:
            print("Error loading GPVAE model:", e)
            model = None
    else:
        model = GPVAE(n_steps=window_size,
                      n_features=n_features,
                      latent_size=tuned_params.get("latent_size", tuning_defaults.get("latent_size", 16)),
                      encoder_sizes=tuned_params.get("encoder_sizes", tuning_defaults.get("encoder_sizes", (64, 32))),
                      decoder_sizes=tuned_params.get("decoder_sizes", tuning_defaults.get("decoder_sizes", (32, 64))),
                      beta=tuned_params.get("beta", tuning_defaults.get("beta", 1.0)),
                      M=tuned_params.get("M", tuning_defaults.get("M", 1)),
                      K=tuned_params.get("K", tuning_defaults.get("K", 1)),
                      kernel=tuned_params.get("kernel", tuning_defaults.get("kernel", "rbf")),
                      sigma=tuned_params.get("sigma", tuning_defaults.get("sigma", 1.0)),
                      length_scale=tuned_params.get("length_scale", tuning_defaults.get("length_scale", 1.0)),
                      kernel_scales=tuned_params.get("kernel_scales", tuning_defaults.get("kernel_scales", 1)),
                      window_size=window_size,
                      batch_size=COMMON_PARAMS["batch_size"],
                      epochs=COMMON_PARAMS["epochs"],
                      patience=COMMON_PARAMS["patience"],
                      num_workers=COMMON_PARAMS["num_workers"],
                      device=COMMON_PARAMS["device"],
                      saving_path=None,
                      model_saving_strategy="best")
        model.fit({"X": train_windows})
        with open(gpvae_model_path, "wb") as f:
            pickle.dump(model, f)
        print("GPVAE training complete and model saved.")
    return model


def load_or_train_forecasting_transformer(window_size, n_features, train_df, forecast_transformer_model_path):
    from data_utils import create_forecast_pairs
    if os.path.exists(forecast_transformer_model_path):
        with open(forecast_transformer_model_path, "rb") as f:
            model = pickle.load(f)
        print("Loaded Forecasting Transformer model from disk.")
    else:
        model = ForecastingTransformer(n_steps=window_size,
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
                                       verbose=False)
        X_train_ft, y_train_ft = create_forecast_pairs(train_df.values, window_size, horizon=1)
        model.fit({"X": X_train_ft, "y": y_train_ft, "X_pred": X_train_ft})
        with open(forecast_transformer_model_path, "wb") as f:
            pickle.dump(model, f)
        print("Forecasting Transformer training complete and model saved.")
    return model
