[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Deep IAQ Imputation & Forecasting Evaluation Framework

Official implementation of the paper:  
**"Evaluating Deep Learning Data Imputation for Subway Indoor Air Quality: Accuracy, Efficiency, and Implications for Downstream Tasks"**  
*Building and Environment, 2025*  
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.buildenv.2025.113713-blue)](https://doi.org/10.1016/j.buildenv.2025.113713)  
[ScienceDirect Link](https://www.sciencedirect.com/science/article/pii/S0360132325011837)

---

This repository provides a modular, extensible framework for benchmarking deep‐learning–based imputation models on indoor air quality (IAQ) time-series data and evaluating their downstream forecasting performance. 

---

## 📂 Repository Structure

- **`config.py`**  
  Global configuration: data paths, model output directories, common training parameters, and rename mappings for plots.

- **`data_utils.py`**  
  Utility functions to:  
  - Load and preprocess IAQ datasets  
  - Generate sliding windows  
  - Reconstruct the full time series from windowed outputs  
  - Post-process model predictions  
  - Produce visualization assets (e.g., heatmaps)

- **`models.py`**  
  Load or train deep‐imputation architectures (US-GAN, Transformer, CSDI, BRITS) and the forecasting transformer.

- **`evaluation.py`**  
  Evaluation routines for:  
  - Traditional imputation baselines  
  - Deep-learning imputation methods under various missingness scenarios  
  - Forecasting accuracy metrics  
  - Missing-data simulation  
  - Automated hyperparameter tuning

- **`main.py`**  
  Pipeline orchestration:  
  1. Ingest and preprocess data  
  2. Train or load models  
  3. Simulate missing-data scenarios  
  4. Evaluate imputation and forecasting  
  5. Export CSV summaries and visualizations (heatmaps, bar charts)

---

> 🔍 **Data Location**  
> The dataset for this evaluation is located in the  
> `analysis_results/` directory.

---

## ⚙️ Dependencies

Ensure you have **Python 3.8+** installed, and include the following in your `requirements.txt`:

```text
pandas
numpy
matplotlib
seaborn
xgboost
torch
pypots

## ✍️ Citation

If you use this repository in your research, please cite:

```bibtex
@article{GHORBANI2025113713,
  title   = {Evaluating Deep Learning Data Imputation for Subway Indoor Air Quality: Accuracy, Efficiency, and Implications for Downstream Tasks},
  author  = {Vahid Ghorbani and Amir Ghorbani and ChangKyoo Yoo},
  journal = {Building and Environment},
  year    = {2025},
  pages   = {113713},
  doi     = {10.1016/j.buildenv.2025.113713}
}
