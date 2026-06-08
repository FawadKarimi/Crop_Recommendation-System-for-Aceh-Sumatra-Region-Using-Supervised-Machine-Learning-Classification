"""
app.py  —  Streamlit Prototype: Crop Recommendation System
MMAI1002 Final Project
---------------------------------------------------------
Run: streamlit run app.py
Requires: crop_model.pkl, scaler.pkl, label_encoder.pkl  (in ./models/)
"""


import sys

def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

if __name__ == "__main__" and not _running_under_streamlit():
    print("This app must be launched with:  streamlit run app.py")
    sys.exit(1)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Recommendation — Aceh/Sumatra",
    page_icon="🌾",
    layout="wide",
)

# ── Load artefacts ─────────────────────────────────────────────────────────────
MDL_DIR = Path(__file__).parent / "models"

@st.cache_resource
def load_model():
    model  = joblib.load(MDL_DIR / "crop_model.pkl")
    scaler = joblib.load(MDL_DIR / "scaler.pkl")
    le     = joblib.load(MDL_DIR / "label_encoder.pkl")
    return model, scaler, le

model, scaler, le = load_model()

# ── Aceh/Sumatra context data (BMKG) ──────────────────────────────────────────
MONTHS        = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
RAINFALL_MM   = [255, 111, 135, 145, 120, 115, 95, 105, 125, 210, 362, 290]
TEMP_AVG      = [26.9,27.1,27.5,27.8,28.0,28.2,27.9,27.8,27.6,27.2,26.8,26.7]

CROP_TIPS = {
    "rice":      "Staple crop of Aceh. Best grown in lowland areas with consistent irrigation.",
    "maize":     "Important food crop in Aceh highlands. Tolerates drier periods.",
    "coconut":   "Widely grown along Aceh's coastline. Thrives in sandy, well-drained soils.",
    "coffee":    "Gayo coffee from Aceh Tengah is globally renowned. Grown at elevation.",
    "oilpalm":   "Largest agricultural contributor to Aceh's GDP (~471,000 ha of plantation).",
    "rubber":    "Common in smallholder farms across Aceh. Tolerates acidic soils.",
    "banana":    "Widely cultivated across Aceh. High humidity suits this crop well.",
    "mango":     "Best in drier months (Jun–Aug) when Aceh experiences less rainfall.",
    "papaya":    "Fast-growing crop, suited to Aceh's warm, humid climate year-round.",
    "sugarcane": "Grown in lowland areas; requires full sun and moderate rainfall.",
    "groundnut": "Good rotation crop for rice fields; tolerates lower soil N.",
    "jute":      "Grown in wet lowlands; benefits from Aceh's heavy monsoon rainfall.",
}

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌾 Crop Recommendation System")
st.markdown(
    "**Aceh / Sumatra Region · MMAI1002 Final Project**  \n"
    "Enter your soil and climate conditions to receive a data-driven crop recommendation."
)
st.divider()

# ── Layout: input | result ─────────────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("Input parameters")
    st.caption("Adjust sliders to match your farm's soil and local climate.")

    subcol1, subcol2 = st.columns(2)
    with subcol1:
        N    = st.slider("Nitrogen (N) — mg/kg",     0,   140,  50,
                         help="Soil nitrogen content")
        P    = st.slider("Phosphorus (P) — mg/kg",   5,   145,  40,
                         help="Soil phosphorus content")
        K    = st.slider("Potassium (K) — mg/kg",    5,   205,  42,
                         help="Soil potassium content")
        ph   = st.slider("Soil pH",                  3.5,  9.0,  6.1, step=0.1)
    with subcol2:
        temp = st.slider("Temperature — °C",         20.0, 35.0, 27.5, step=0.1,
                         help="Banda Aceh avg: 26.7–28.2 °C (BMKG)")
        hum  = st.slider("Humidity — %",             30.0, 100.0, 79.0, step=0.5,
                         help="Banda Aceh avg: ~79–81% (BMKG)")
        rain = st.slider("Rainfall — mm/month",      20.0, 380.0, 150.0, step=1.0,
                         help="Banda Aceh: 95–362 mm/month (BMKG)")

    st.divider()
    predict_btn = st.button("Recommend crop  →", type="primary", use_container_width=True)

with col_result:
    st.subheader("Recommendation")
    if predict_btn:
        X_in = np.array([[N, P, K, temp, hum, ph, rain]])
        X_sc = scaler.transform(X_in)
        pred_enc  = model.predict(X_sc)[0]
        probas    = model.predict_proba(X_sc)[0]
        crop      = le.inverse_transform([pred_enc])[0]
        confidence = probas[pred_enc] * 100

        st.success(f"**Recommended: {crop.upper()}**", icon="✅")
        st.metric("Model confidence", f"{confidence:.1f}%")

        tip = CROP_TIPS.get(crop, "")
        if tip:
            st.info(f"**About this crop in Aceh:** {tip}")

        # Top-5 alternatives
        top5_idx  = np.argsort(probas)[::-1][:5]
        top5_data = {
            "Crop":        [le.classes_[i] for i in top5_idx],
            "Probability": [f"{probas[i]*100:.1f}%" for i in top5_idx],
        }
        st.markdown("**Top 5 predictions:**")
        st.dataframe(pd.DataFrame(top5_data), hide_index=True, use_container_width=True)

        # Probability bar chart
        fig, ax = plt.subplots(figsize=(5, 2.5))
        top_labels = [le.classes_[i] for i in top5_idx]
        top_probs  = [probas[i] for i in top5_idx]
        bars = ax.barh(top_labels[::-1], top_probs[::-1], color="#1D9E75")
        for bar, val in zip(bars, top_probs[::-1]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{val*100:.1f}%", va="center", fontsize=8)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Probability")
        ax.set_title("Top 5 crop probabilities", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Adjust the parameters on the left and click **Recommend crop**.")

# ── Aceh Climate Context ───────────────────────────────────────────────────────
st.divider()
st.subheader("Aceh climate context (BMKG data)")
st.caption(
    "Monthly climate normals for Banda Aceh. "
    "Source: BMKG (dataonline.bmkg.go.id) via climatestotravel.com"
)

fig2, ax1 = plt.subplots(figsize=(11, 4))
ax2 = ax1.twinx()
ax1.bar(MONTHS, RAINFALL_MM, color="#5DCAA5", alpha=0.75, label="Rainfall (mm)", edgecolor="white")
ax1.set_ylabel("Monthly rainfall (mm)", color="#0F6E56")
ax1.tick_params(axis="y", labelcolor="#0F6E56")
ax2.plot(MONTHS, TEMP_AVG, "o-", color="#D85A30", linewidth=2, markersize=5, label="Avg temp (°C)")
ax2.set_ylabel("Average temperature (°C)", color="#D85A30")
ax2.tick_params(axis="y", labelcolor="#D85A30")
ax2.set_ylim(24, 32)
ax1.set_ylim(0, max(RAINFALL_MM) * 1.3)
import matplotlib.patches as mpatches
handles = [
    mpatches.Patch(color="#5DCAA5", label="Rainfall (mm)"),
    plt.Line2D([0],[0], color="#D85A30", marker="o", label="Avg temperature (°C)"),
]
ax1.legend(handles=handles, loc="upper left", fontsize=9)
ax1.set_title("Banda Aceh monthly climate normals", fontsize=11)
plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.caption(
    "**Why this matters:** Aceh's equatorial climate — annual rainfall ~2,315 mm, "
    "temperature 26.7–28.2 °C, humidity ~79–81% — strongly influences which crops "
    "are viable. The ML model was trained on feature distributions calibrated to "
    "these regional climate norms."
)
