# F1 Podium Prediction Project  
**Predicting Formula 1 podium finishes (2012–2025) using regularized logistic regression**

![F1 car speeding on track  
*(Photo by Patrick Robert Doyle on Unsplash)*

## Executive Summary  
**Bottom line, up front:**  
**Starting grid position beats everything else — by a mile.**  

A dead-simple baseline that just predicts “the top-3 qualifiers will podium” already achieves **~75% podium precision** on both the 2012–2024 hold-out data and the full 2025 out-of-sample season. After exhaustive feature engineering and state-of-the-art modeling (Elastic Net + interactions + bootstrap CIs), no combination of driver age, nationality, team, circuit characteristics, or pit strategy meaningfully outperforms simply using grid position.

## Key Findings
- A **single grid place gained** is worth dramatically more podium probability than any other variable we measured.
- Best model (Elastic Net + lat × British interaction) → 84.3% overall accuracy, 84.9% AUC on held-out data.
-  - Statistically credible positive effects: no pit stops, Mercedes, Red Bull, Ferrari, longer stints (laps_per_pit)
  - Strongest negative effect by far: `grid` (β ≈ –0.45, CI entirely negative)
- On the 2025 season (21 races as of Nov 19, 2025), the model correctly identifies 33% of actual podiums vs. 74.6% for the pure-grid baseline.
- Conclusion: The team that qualifies highest on average in 2025 will almost certainly win the Constructors’ Championship.

## Data
- Primary source: Ergast Developer API (Kaggle mirror) – all races 1950–2024
- 2025 season: manually curated results up to Brazilian GP (Nov 9, 2025)
- Final dataset used: 2012–2024 (5,524 driver-race observations)
- Features: grid, driver age, nationality & constructor dummies, circuit lat/lng/alt, pit-stop summary stats, one interaction term

## Models & Methods
- Baseline logistic regression
- Ridge, LASSO, Elastic Net (all with 100-iteration nonparametric bootstrap CIs)
- Interaction term search + exhaustive subset evaluation
- Principal Component Regression (PCR) for comparison
- Decision-theoretic evaluation with asymmetric loss function

## Repository Contents
- `notebooks/F1_Podium_Prediction.ipynb` – full reproducible analysis (Plan → Build → Explore → Model → Predict → Communicate)
- `data/` – cleaned CSVs and 2025 season file
