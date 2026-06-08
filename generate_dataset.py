"""
generate_dataset.py
--------------------
Generates a realistic Aceh/Sumatra-calibrated crop recommendation dataset.

Real BMKG/climate statistics used to set feature distributions:
  - Banda Aceh avg temp    : 26.9 – 31.2 °C  (source: BMKG / climatestotravel.com)
  - Banda Aceh avg humidity: 79 – 81%         (source: climatestotravel.com)
  - Annual rainfall        : ~2315 mm/year    (source: weather-and-climate.com)
  - Monthly range          : 111 mm (Feb) – 362 mm (Nov)
  - Sumatra avg temp       : 27 °C annual avg (source: worlddata.info)

Target crops selected for regional relevance to Aceh/Sumatra agriculture:
  rice, maize, coconut, coffee (arabica), oil palm, rubber, banana,
  mango, papaya, sugarcane, groundnut, jute
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng  = np.random.default_rng(SEED)

CROPS = {
    # crop : (N_mean, N_std, P_mean, P_std, K_mean, K_std,
    #         temp_mean, temp_std, hum_mean, hum_std,
    #         ph_mean, ph_std, rain_mean, rain_std, n_samples)
    # All distributions calibrated to tropical Aceh/Sumatra ranges
    "rice":       (80,  10,  40,  8,  40,  8,  27,  1.2, 82, 3, 6.5, 0.3, 230, 40, 220),
    "maize":      (80,  12,  48, 10,  50, 10,  27,  1.5, 65, 5, 6.0, 0.4, 100, 20, 180),
    "coconut":    (22,   6,  16,  5,  30,  7,  29,  1.0, 82, 3, 6.0, 0.3, 200, 30, 180),
    "coffee":     (18,   5,  42,  8,  43,  8,  24,  1.5, 72, 4, 6.3, 0.3, 155, 25, 180),
    "oilpalm":    (30,   8,  35,  8,  45,  8,  28,  1.0, 80, 3, 5.5, 0.4, 220, 35, 200),
    "rubber":     (25,   7,  30,  7,  40,  8,  28,  1.2, 80, 4, 5.8, 0.4, 200, 30, 180),
    "banana":     (85,  12,  75, 12,  50, 10,  28,  1.2, 80, 4, 6.0, 0.3, 110, 20, 160),
    "mango":      (22,   6,  20,  6,  30,  7,  30,  1.0, 50, 5, 6.5, 0.4,  90, 20, 140),
    "papaya":     (50,  10,  59,  9,  50,  8,  28,  1.2, 90, 3, 6.5, 0.3,  90, 15, 140),
    "sugarcane":  (20,   6,  20,  6,  20,  6,  29,  1.0, 65, 5, 6.5, 0.3,  85, 15, 140),
    "groundnut":  (25,   6,  70, 10,  55, 10,  28,  1.2, 65, 5, 5.8, 0.4, 100, 20, 140),
    "jute":       (72,  10,  46,  8,  44,  8,  28,  1.5, 80, 4, 6.0, 0.4, 175, 25, 140),
}

rows = []
for crop, params in CROPS.items():
    (N_m,N_s,P_m,P_s,K_m,K_s,T_m,T_s,H_m,H_s,ph_m,ph_s,R_m,R_s,n) = params
    N    = np.clip(rng.normal(N_m,  N_s,  n),   0, 140)
    P    = np.clip(rng.normal(P_m,  P_s,  n),   5, 145)
    K    = np.clip(rng.normal(K_m,  K_s,  n),   5, 205)
    temp = np.clip(rng.normal(T_m,  T_s,  n),  20,  44)
    hum  = np.clip(rng.normal(H_m,  H_s,  n),  14, 100)
    ph   = np.clip(rng.normal(ph_m, ph_s, n),   3,   9)
    rain = np.clip(rng.normal(R_m,  R_s,  n),  20, 350)
    for i in range(n):
        rows.append({
            "N": round(N[i], 1),
            "P": round(P[i], 1),
            "K": round(K[i], 1),
            "temperature": round(temp[i], 2),
            "humidity":    round(hum[i],  2),
            "ph":          round(ph[i],   2),
            "rainfall":    round(rain[i], 2),
            "label":       crop,
        })

df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)

out = Path(__file__).parent / "aceh_crop_data.csv"
df.to_csv(out, index=False)
print(f"Dataset saved -> {out}")
print(f"Shape: {df.shape}")
print(df["label"].value_counts().to_string())
print(f"\nFeature stats (sample):")
print(df.describe().round(2).to_string())
