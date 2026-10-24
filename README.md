# Crop Recommendation for Aceh & Sumatra (Streamlit Prototype)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Classification-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

Supervised **crop recommendation** prototype for the **Aceh / Sumatra** agro-climatic context. Given soil nutrients and weather features, the system suggests suitable crops using trained **Decision Tree**, **Random Forest**, and **KNN** models—with a **Streamlit** front end for interactive exploration.

> **Scope:** Course research prototype (**MMAI1002**). Uses **region-calibrated synthetic data** for demonstration. **Not field-validated** for commercial agronomy decisions.

## Demo

```bash
git clone https://github.com/FawadKarimi/Crop_Recommendation-System-for-Aceh-Sumatra-Region-Using-Supervised-Machine-Learning-Classification.git
cd Crop_Recommendation-System-for-Aceh-Sumatra-Region-Using-Supervised-Machine-Learning-Classification
pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib
streamlit run app.py
```

Pre-trained artifacts in `models/` are loaded automatically (`crop_model.pkl`, `scaler.pkl`, `label_encoder.pkl`).

## Project layout

```
├── app.py                 # Streamlit UI
├── train.py               # Training, CV, figures, model export
├── eda.py                 # Exploratory analysis scripts
├── generate_dataset.py    # Synthetic Aceh/Sumatra-style dataset generator
├── aceh_crop_data.csv     # Tabular training data
├── models/                # Saved model, scaler, encoder, metrics JSON
├── figures/               # Manuscript-ready plots (confusion matrix, radar, etc.)
└── crop_recommendation_manuscript.docx
```

## Model performance (validation snapshot)

From `models/results_summary.json` (held-out validation + 5-fold CV):

| Model | Val accuracy | Val F1 | CV mean | CV std |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | 0.850 | 0.850 | 0.815 | 0.018 |
| **Random Forest** | **0.897** | **0.894** | **0.886** | **0.017** |
| KNN (k=5) | 0.870 | 0.866 | 0.884 | 0.011 |

**Best model on validation:** Random Forest (exported as `models/crop_model.pkl`).

## Features used

Typical inputs include **N, P, K**, **temperature**, **humidity**, **pH**, and **rainfall** (see CSV headers). Radar and correlation plots in `figures/` document crop-specific profiles and feature relationships for the Aceh climate narrative.

## Retrain from scratch

```bash
python generate_dataset.py   # optional: refresh synthetic data
python eda.py                # optional: regenerate EDA figures
python train.py              # trains DT/RF/KNN, writes models/ + figures/
streamlit run app.py
```

## Dev container

A `.devcontainer/devcontainer.json` is included for cloud IDE workflows (installs dependencies and launches Streamlit on port **8501**).

## Limitations

- Synthetic / curated data — **not** a substitute for soil labs or local extension services.  
- Class balance and label definitions follow the course dataset design.  
- Always validate recommendations with **local agronomists** before operational use.

## Author

**Mohammad Fawad Karimi** — Universitas Syiah Kuala  
📧 [mfkarimi@mhs.usk.ac.id](mailto:mfkarimi@mhs.usk.ac.id) · 🌐 [Portfolio](https://fawadkarimi.github.io/)
