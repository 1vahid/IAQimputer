# main.py
import os
import time                                       # ← optional: if used elsewhere
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

from config import (
    COMMON_PARAMS, RENAME_MAP, OPT_DIR,
    FORECAST_TRANSFORMER_MODEL_PATH,
    MODEL_DIR, DATA_DIR, TUNING_EPOCHS
)
from data_utils import (
    load_and_prepare_data, create_windows,
    generate_heatmap, reconstruct_series,
    process_prediction_result
)
from models import (
    train_or_load_usgan, train_or_load_transformer,
    train_or_load_csdi, train_or_load_saits,
    train_or_load_stemgnn, train_or_load_brits,
    train_or_load_frets, train_or_load_micn,
    train_or_load_gpvae, load_or_train_forecasting_transformer
)
from evaluation import (
    forecast_evaluation, forecast_transformer_eval,
    evaluate_imputation_full, evaluate_imputation_windowed,
    simulate_MCAR, simulate_MAR, simulate_MNAR,
    simulate_block_missingness, create_forecast_pairs,
    advanced_tune_model
)

def main():
    # ---------------------------
    # Data Preparation
    # ---------------------------
    (train_df_orig, train_df, val_df, test_df, test_df_orig,
     train_mean, train_std, selected_cols) = load_and_prepare_data()

    window_size       = COMMON_PARAMS["window_size"]
    train_windows     = create_windows(train_df, window_size)
    val_windows       = create_windows(val_df, window_size)
    test_windows      = create_windows(test_df, window_size)
    test_windows_orig = create_windows(test_df_orig, window_size)
    n_features        = len(selected_cols)

    # ---------------------------
    # Advanced Hyperparameter Tuning
    # ---------------------------
    tuned_params_file = os.path.join(OPT_DIR, "tuned_hyperparameters_summary.csv")
    if os.path.exists(tuned_params_file):
        tuned_params_df = pd.read_csv(tuned_params_file, index_col=0)
        tuned_params    = tuned_params_df.to_dict(orient="index")
        print("Loaded tuned hyperparameters from file:")
        print(tuned_params_df)
    else:
        print("Tuned hyperparameters file not found; running tuning procedure...")
        tuned_params = {}

        tuning_grids = {
             "USGAN": {
                 "rnn_hidden_size": [64, 128],
                 "lambda_mse": [0.05, 0.1],
                 "hint_rate": [0.1, 0.2],
                 "dropout": [0.1, 0.2]
             },
           "StemGNN": {
                "n_layers": [2, 3],
                "d_model": [64, 128],
                "dropout": [0.1, 0.2],
                "n_stacks": [1, 2]
            },
            "Transformer": {
                "n_layers": [2, 3, 4],
                "d_model": [64, 128, 256],
                "n_heads": [4, 8],
                "d_k": [16, 32],
                "d_v": [16, 32],
                "d_ffn": [128, 256, 512],
                "dropout": [0.1, 0.2],
                "attn_dropout": [0.1, 0.2]
            },
            "CSDI": {
                "n_layers": [2, 3, 4],
                "n_heads": [4, 8],
                "n_channels": [64, 128],
                "d_time_embedding": [16, 32],
                "d_feature_embedding": [16, 32],
                "d_diffusion_embedding": [16, 32]
            },
            "SAITS": {
                "n_layers": [2, 3, 4],
                "d_model": [128, 256],
                "n_heads": [4, 8],
                "d_k": [64, 128],
                "d_v": [64, 128],
                "d_ffn": [128, 256],
                "dropout": [0.1, 0.2],
                "attn_dropout": [0.1, 0.2]
            },

            "BRITS": {
                "rnn_hidden_size": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
            },
            "FreTS": {
                "embed_size": [32, 64, 128],
                "hidden_size": [256, 512],
                "channel_independence": [True, False]
            },
            "MICN": {
                "n_layers": [2, 3, 4, 5],
                "d_model": [64, 128, 512],
                "conv_kernel": [[3]]
            },
            "GPVAE": {
                "latent_size": [16, 32],
                "encoder_sizes": [(64, 32), (128, 64)],
                "decoder_sizes": [(32, 64), (64, 128)],
                "beta": [1.0, 2.0],
                "M": [1],
                "K": [1],
                "sigma": [1.0, 2.0],
                "length_scale": [1.0, 2.0],
                "kernel_scales": [1, 2]
            }

        }



        from pypots.imputation.transformer.model import Transformer as ImputationTransformer
        from pypots.imputation.csdi.model import CSDI
        from pypots.imputation.saits.model import SAITS
        from pypots.imputation.stemgnn.model import StemGNN
        from pypots.imputation.brits.model import BRITS
        from pypots.imputation.frets.model import FreTS
        from pypots.imputation.micn.model import MICN
        from pypots.imputation.gpvae.model import GPVAE
        from pypots.imputation.usgan.model import USGAN

        model_mapping = {
            "USGAN": USGAN,
            "StemGNN": StemGNN,
            "Transformer": ImputationTransformer,
            "CSDI": CSDI,
            "SAITS": SAITS,
            "BRITS": BRITS,
            "FreTS": FreTS,
            "MICN": MICN,
            "GPVAE": GPVAE
        }

        for model_name in tuning_grids:
            print(f"Tuning {model_name} …")
            if model_name not in model_mapping:
                print(f"Skipping {model_name} (no class).")
                continue
            best, results = advanced_tune_model(
                model_mapping[model_name],
                tuning_grids[model_name],
                train_windows, val_windows,
                model_name, COMMON_PARAMS,
                n_iter=10, missing_rate=0.3,
                train_std=train_std, train_mean=train_mean
            )
            tuned_params[model_name] = best
            pd.DataFrame(results).to_csv(
                os.path.join(OPT_DIR, f"{model_name}_tuning_results.csv"),
                index=False
            )
            print(f"Best {model_name} params: {best}")

        pd.DataFrame.from_dict(tuned_params, orient="index")\
          .to_csv(tuned_params_file)
        print("Tuned Hyperparameters Summary saved.")

    # ---------------------------
    # Load or Train Deep Models
    # ---------------------------
    usgan_model        = train_or_load_usgan(
        window_size, n_features, train_windows,
        os.path.join(MODEL_DIR, "usgan_model.pkl")
    )
    transformer_model  = train_or_load_transformer(
        window_size, n_features, train_windows,
        os.path.join(MODEL_DIR, "transformer_model.pkl"),
        tuned_params.get("Transformer", {}),
        {"n_layers":2,"d_model":64,"n_heads":4,"d_k":16,"d_v":16,"d_ffn":128,"dropout":0.1,"attn_dropout":0.1}
    )
    csdi_model         = train_or_load_csdi(
        window_size, n_features, train_windows,
        os.path.join(MODEL_DIR, "csdi_model.pkl"),
        tuned_params.get("CSDI", {}),
        {"n_layers":2,"n_heads":4,"n_channels":64,"d_time_embedding":16,"d_feature_embedding":16,"d_diffusion_embedding":16}
    )
    try:
        saits_model    = train_or_load_saits(
            window_size, n_features, train_windows,
            os.path.join(MODEL_DIR, "saits_model.pkl"),
            tuned_params.get("SAITS", {}),
            {"n_layers":2,"d_model":128,"n_heads":4,"d_k":64,"d_v":64,"d_ffn":128,"dropout":0.1,"attn_dropout":0.1}
        )
    except Exception as e:
        print("SAITS failed:", e)
        saits_model    = None

    try:
        stemgnn_model = train_or_load_stemgnn(
            window_size, n_features, train_windows,
            os.path.join(MODEL_DIR, "stemgnn_model.pkl"),
            tuned_params.get("StemGNN", {}),
            {"n_layers":2,"d_model":64,"dropout":0.1}
        )
    except Exception as e:
        print("StemGNN failed:", e)
        stemgnn_model  = None

    brits_model       = train_or_load_brits(
        window_size, n_features, train_windows,
        os.path.join(MODEL_DIR, "brits_model.pkl")
    )

    try:
        frets_model    = train_or_load_frets(
            window_size, n_features, train_windows,
            os.path.join(MODEL_DIR, "frets_model.pkl"),
            tuned_params.get("FreTS", {}),
            {"embed_size":128,"hidden_size":256,"channel_independence":False}
        )
    except Exception as e:
        print("FreTS failed:", e)
        frets_model    = None

    try:
        micn_model     = train_or_load_micn(
            window_size, n_features, train_windows,
            os.path.join(MODEL_DIR, "micn_model.pkl"),
            tuned_params.get("MICN", {}),
            {"n_layers":2,"d_model":64,"conv_kernel":[3]}
        )
    except Exception as e:
        print("MICN failed:", e)
        micn_model     = None

    try:
        gpvae_model    = train_or_load_gpvae(
            window_size, n_features, train_windows,
            os.path.join(MODEL_DIR, "gpvae_model.pkl"),
            tuned_params.get("GPVAE", {}),
            {"latent_size":16,"encoder_sizes":(64,32),"decoder_sizes":(32,64),
             "beta":1.0,"M":1,"K":1,"sigma":1.0,"length_scale":1.0,"kernel_scales":1}
        )
    except Exception as e:
        print("GPVAE failed:", e)
        gpvae_model    = None

    print("Loading/Training Forecasting Transformer…")
    forecasting_transformer = load_or_train_forecasting_transformer(
        window_size, n_features, train_df,
        FORECAST_TRANSFORMER_MODEL_PATH
    )

    # ---------------------------
    # Create Missing Data Scenarios
    # ---------------------------
    scenarios_test = {}
    for rate in [0.1, 0.2, 0.3, 0.5, 0.9]:
        key = rf"$\text{{MCAR}}_{{{int(rate*100)}\%}}$"
        scenarios_test[key] = simulate_MCAR(test_df.copy(), missing_prob=rate)

    for rate in [0.3, 0.5]:
        k_l = rf"$\text{{MAR Lobby PM}}_{{2.5\,({int(rate*100)}\%)}}$"
        k_p = rf"$\text{{MAR Platform PM}}_{{2.5\,({int(rate*100)}\%)}}$"
        scenarios_test[k_l] = simulate_MAR(
            test_df.copy(), "Lobby_PM2_5_Indoor",
            threshold_quantile=0.2, missing_prob=rate
        )
        scenarios_test[k_p] = simulate_MAR(
            test_df.copy(), "Platform_PM2_5_Indoor",
            threshold_quantile=0.2, missing_prob=rate
        )

    for rate in [0.3, 0.5]:
        k_m = rf"$\text{{MNAR Lobby PM}}_{{10\,({int(rate*100)}\%)}}$"
        scenarios_test[k_m] = simulate_MNAR(
            test_df.copy(), "Lobby_PM10_Indoor",
            high_quantile=0.8, missing_prob=rate
        )

    block_length = int(0.10 * test_df.shape[0])
    block_data, _, _ = simulate_block_missingness(
        test_df.copy(), block_length=block_length
    )
    scenarios_test[r"$\text{Block All}$"] = block_data

    # ---------------------------
    # Prompt User for Imputation Methods
    # ---------------------------
    available_imp_methods = [
        "mean","median","linear","usgan","transformer",
        "csdi","saits","stemgnn","brits","frets","micn","gpvae"
    ]
    sel = input(
        "Enter comma-separated imputation methods "
        f"(choices: {', '.join(available_imp_methods)}): "
    )
    selected_imp_methods = [m.strip().lower() for m in sel.split(",") if m.strip()]
    if not selected_imp_methods:
        print("No valid methods selected; defaulting to all.")
        selected_imp_methods = available_imp_methods
    print("Selected methods:", selected_imp_methods)

    # ---------------------------
    # Missing Scenarios Summary & Heatmap
    # ---------------------------
    summary_list = []
    for name, df_scn in scenarios_test.items():
        perc = df_scn.isnull().mean() * 100
        row = {"Scenario": name}
        for col in df_scn.columns:
            row[col] = round(perc[col], 2)
        summary_list.append(row)

    summary_df = pd.DataFrame(summary_list).set_index("Scenario")
    summary_df.rename(columns=RENAME_MAP, inplace=True)
    print("Missing Data Scenarios Summary:")
    print(summary_df)

    os.makedirs("analysis_results", exist_ok=True)
    summary_df.to_csv(
        os.path.join("analysis_results","missing_data_scenarios_summary_test.csv")
    )
    generate_heatmap(
        summary_df,
        "Missing Data Percentage per Scenario",
        os.path.join("analysis_results","missing_scenarios_heatmap.png")
    )

    # ---------------------------
    # Imputation Evaluation Across Scenarios
    # ---------------------------
    imputation_results_list = []
    for scenario_name, scenario_df in scenarios_test.items():
        rd = {"Scenario": scenario_name}
        # Traditional
        for meth in ["mean","median","linear"]:
            if meth in selected_imp_methods:
                mae, mre = evaluate_imputation_full(
                    test_df, scenario_df,
                    method=meth,
                    train_df=train_df,
                    train_std=train_std,
                    train_mean=train_mean
                )
                rd[f"MAE_{meth.capitalize()}"] = round(mae,4)
                rd[f"MRE_{meth.capitalize()}"] = round(mre,4)

        # Deep + timing
        sim_windows = create_windows(scenario_df, window_size)
        for meth, model in [
            ("USGAN",      usgan_model),
            ("TRANSFORMER",transformer_model),
            ("CSDI",       csdi_model),
            ("SAITS",      saits_model),
            ("STEMGNN",    stemgnn_model),
            ("BRITS",      brits_model),
            ("FRETS",      frets_model),
            ("MICN",       micn_model),
            ("GPVAE",      gpvae_model),
        ]:
            if meth.lower() not in selected_imp_methods or model is None:
                continue
            mae, mre, inf_t = evaluate_imputation_windowed(
                test_windows_orig, sim_windows,
                model, train_std, train_mean
            )
            rd[f"MAE_{meth}"]  = round(mae,4)
            rd[f"MRE_{meth}"]  = round(mre,4)
            rd[f"Time_{meth}"] = round(inf_t,4)

        imputation_results_list.append(rd)

    imputation_results_df = pd.DataFrame(imputation_results_list).set_index("Scenario")
    print("\nImputation Evaluation Results:")
    print(imputation_results_df)

    imp_csv = os.path.join("analysis_results","imputation_evaluation_results.csv")
    imputation_results_df.to_csv(imp_csv)
    print(f"Saved imputation+timings to {imp_csv}")

    # Save timing-only CSV
    time_cols = [c for c in imputation_results_df.columns if c.startswith("Time_")]
    timing_df = imputation_results_df[time_cols]
    timing_csv = os.path.join("analysis_results","inference_time_results.csv")
    timing_df.to_csv(timing_csv)
    print(f"Saved inference times to {timing_csv}")

    # ---------------------------
    # Prompt User for Forecasting Evaluation
    # ---------------------------
    sel_f = input(
        "Enter comma-separated methods for forecasting (MCAR_50%): "
        f"(choices: {', '.join(available_imp_methods)}): "
    )
    selected_forecast_methods = [m.strip().lower() for m in sel_f.split(",") if m.strip()]
    if not selected_forecast_methods:
        selected_forecast_methods = available_imp_methods
    print("Selected forecasting methods:", selected_forecast_methods)

    # ---------------------------
    # Downstream Forecasting Evaluation (MCAR_50%)
    # ---------------------------
    forecast_key = rf"$\text{{MCAR}}_{{50\%}}$"
    if forecast_key not in scenarios_test:
        raise ValueError("MCAR_50% not defined.")
    scenario_50 = scenarios_test[forecast_key]

    forecast_results = {}
    for method in selected_forecast_methods:
        if method in ["mean","median","linear"]:
            df_c = scenario_50.copy()
            if method=="mean":
                for col in df_c: df_c[col]=df_c[col].fillna(train_df[col].mean())
            elif method=="median":
                for col in df_c: df_c[col]=df_c[col].fillna(train_df[col].median())
            else:
                df_c = df_c.interpolate(method="linear", limit_direction="both")
            imputed_denorm = df_c * train_std + train_mean
        else:
            sim_w = create_windows(scenario_50, window_size)
            model = {
                **{"usgan":usgan_model,"transformer":transformer_model,
                   "csdi":csdi_model,"saits":saits_model,
                   "stemgnn":stemgnn_model,"brits":brits_model,
                   "frets":frets_model,"micn":micn_model,
                   "gpvae":gpvae_model}
            }.get(method)
            if model is None: continue
            res = model.predict({"X":sim_w})
            pred = process_prediction_result(res)
            if pred.ndim==4: pred = pred.mean(axis=1)
            if pred.shape[-1]!=train_df.shape[1]:
                pred = pred[...,:train_df.shape[1]]
            rec = reconstruct_series(pred, window_size, len(test_df_orig))
            imputed_denorm = pd.DataFrame(rec, index=test_df_orig.index, columns=test_df_orig.columns)

        mae_xgb, _ = forecast_evaluation(imputed_denorm, train_df_orig, test_df_orig, window_size)
        mae_ft     = forecast_transformer_eval(
            forecasting_transformer,
            imputed_denorm.values,
            window_size, horizon=1,
            train_mean=train_mean, train_std=train_std
        )
        forecast_results[method] = {
            "XGBoost_MAE": round(mae_xgb,4),
            "Transformer_MAE": round(mae_ft,4)
        }

    forecast_results_df = pd.DataFrame.from_dict(forecast_results, orient="index")
    print("\nDownstream Forecasting Evaluation Results:")
    print(forecast_results_df)
    forecast_results_csv = os.path.join("analysis_results","forecasting_evaluation_results.csv")
    forecast_results_df.to_csv(forecast_results_csv)

    # ---------------------------
    # Plot Forecasting Results
    # ---------------------------
    fig, ax = plt.subplots(figsize=(12,7))
    methods_list = forecast_results_df.index.tolist()
    x = np.arange(len(methods_list))
    w = 0.35

    bars1 = ax.bar(x - w/2, forecast_results_df["XGBoost_MAE"], w, label="XGBoost")
    bars2 = ax.bar(x + w/2, forecast_results_df["Transformer_MAE"], w, label="Transformer")

    ax.set_xlabel("Imputation Method", fontsize=14, fontname="Times New Roman")
    ax.set_ylabel("Forecast MAE", fontsize=14, fontname="Times New Roman")
    ax.set_title("Downstream Forecasting on MCAR_50%", fontsize=16, fontname="Times New Roman")
    ax.set_xticks(x)
    ax.set_xticklabels(methods_list, rotation=45, fontsize=12, fontname="Times New Roman")
    ax.legend(fontsize=12, prop={"family": "Times New Roman"})

    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}",
                    xy=(bar.get_x()+bar.get_width()/2, h),
                    xytext=(0,3), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=12, color="black", fontname="Times New Roman")

    plt.tight_layout()
    fig_path = os.path.join("analysis_results","forecasting_evaluation.png")
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.show()

    print(f"Forecast CSV → {forecast_results_csv}")
    print(f"Forecast figure → {fig_path}")
    print("\nAll evaluations completed.")

if __name__ == "__main__":
    main()
