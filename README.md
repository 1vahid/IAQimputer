[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Made with ❤️ for Research](https://img.shields.io/badge/Made%20with-%E2%9D%A4-red)](#)

# 🌱 Deep IAQ Imputation & Forecasting Evaluation Framework  

Official implementation of the paper:  
**"Evaluating Deep Learning Data Imputation for Subway Indoor Air Quality: Accuracy, Efficiency, and Implications for Downstream Tasks"**  
*Building and Environment, 2025*  

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.buildenv.2025.113713-blue)](https://doi.org/10.1016/j.buildenv.2025.113713)  
[📖 ScienceDirect Link](https://www.sciencedirect.com/science/article/pii/S0360132325011837)

---

This repository provides a **modular, extensible, and research-ready framework** for benchmarking deep-learning–based **imputation** models on indoor air quality (IAQ) time-series data and evaluating their **downstream forecasting** performance.  

🔥 Out of the box, you can:  
- Benchmark **9 state-of-the-art deep models**  
- Run **traditional statistical baselines** for comparison  
- Simulate **missingness scenarios** (MCAR, MAR, MNAR, block)  
- Evaluate **forecasting accuracy** on reconstructed IAQ data  


---

## 📂 Repository Structure

```
├── config.py         # Global config: paths, model dirs, parameters
├── data_utils.py     # Data loading, preprocessing, sliding windows, heatmaps
├── models.py         # Deep models (US-GAN, Transformer, CSDI, BRITS, etc.)
├── evaluation.py     # Baselines, metrics, missingness simulation, tuning
├── main.py           # Orchestrates the full pipeline
└── analysis_results/ # Place your IAQ dataset here
```

---

## 🤖 Models Implemented

| Model Name   | Description |
|--------------|-------------|
| **BRITS**    | Bidirectional Recurrent Imputation for Time Series |
| **SAITS**    | Self-Attention-based Imputation for Time Series |
| **Transformer** | Self-attention for temporal dependencies |
| **FreTS**    | Frequency-enhanced Transformer for irregular time-series |
| **MICN**     | Multi-scale Interpolation Convolutional Network |
| **CSDI**     | Conditional Score-based Diffusion Imputation |
| **US-GAN**   | Unsupervised GAN for time-series imputation |
| **GP-VAE**   | Gaussian Process Prior Variational Autoencoder |
| **USGAN**    | GAN baseline for irregular time-series |

✔️ **All nine models supported:**  
SAITS · Transformer · BRITS · MICN · FreTS · CSDI · US-GAN · USGAN · GP-VAE  

---

## ⚡ Quick Start

### 1️⃣ Clone the repo
```bash
git clone https://github.com/1vahid/IAQimputer.git
cd IAQimputer
```

### 2️⃣ Install dependencies
Requires **Python 3.8+**:
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the pipeline
```bash
python main.py
```

👉 This will:  
1. Load & preprocess IAQ data  
2. Train or load the chosen model  
3. Simulate missingness (e.g., MAR)  
4. Evaluate imputation & forecasting  
5. Save results in `analysis_results/`  

---

## ⚙️ Dependencies

Core stack (pinned in `requirements.txt`):  

- `pandas`, `numpy` – data handling  
- `matplotlib`, `seaborn` – visualization  
- `torch` – deep learning backend  
- `pypots` – time-series imputation models  
- `xgboost` – forecasting benchmark  

---

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
```

---

🚀 *Happy Researching!*  
