"""
Generates a synthetic training set for the Analytical Node.

Sampling strategy: draw features from realistic agronomic ranges (not uniform
noise) so the model sees plausible combinations, including correlated wet
conditions (humidity+rainfall often co-move) and independent NPK deficiency
cases. Add small Gaussian label noise so XGBoost has to genuinely learn a
smooth function rather than memorize a deterministic formula.
"""

import numpy as np
import pandas as pd
from risk_heuristic import compute_risk_components, risk_band

RNG_SEED = 42
N_SAMPLES = 6000
LABEL_NOISE_STD = 3.0  # percentage points of noise on final risk_pct


def generate(n_samples: int = N_SAMPLES, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    temperature_c = rng.normal(24, 7, n_samples).clip(5, 45)

    # humidity and rainfall are positively correlated in reality (wet weather -> humid air)
    base_wetness = rng.beta(2, 2, n_samples)  # 0-1 latent "wetness" factor
    humidity_pct = (40 + base_wetness * 55 + rng.normal(0, 6, n_samples)).clip(10, 100)
    rainfall_mm_14d = (base_wetness * 90 + rng.normal(0, 12, n_samples)).clip(0, 150)

    soil_n = rng.normal(85, 35, n_samples).clip(5, 180)
    soil_p = rng.normal(38, 18, n_samples).clip(2, 90)
    soil_k = rng.normal(55, 25, n_samples).clip(5, 140)

    soil_moisture_pct = rng.normal(48, 16, n_samples).clip(5, 95)

    rows = []
    for i in range(n_samples):
        comp = compute_risk_components(
            temperature_c[i], humidity_pct[i], rainfall_mm_14d[i],
            soil_n[i], soil_p[i], soil_k[i], soil_moisture_pct[i],
        )
        noisy_risk = float(np.clip(comp["risk_pct"] + rng.normal(0, LABEL_NOISE_STD), 0, 100))
        rows.append({
            "temperature_c": temperature_c[i],
            "humidity_pct": humidity_pct[i],
            "rainfall_mm_14d": rainfall_mm_14d[i],
            "soil_n": soil_n[i],
            "soil_p": soil_p[i],
            "soil_k": soil_k[i],
            "soil_moisture_pct": soil_moisture_pct[i],
            "risk_pct": noisy_risk,
            "risk_band": risk_band(noisy_risk),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv("synthetic_environmental_risk.csv", index=False)
    print(f"Generated {len(df)} rows -> synthetic_environmental_risk.csv")
    print(df.describe())
    print("\nRisk band distribution:")
    print(df["risk_band"].value_counts())
