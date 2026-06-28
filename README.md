# Task-1 — Smart Restaurant Rating Predictor

<p>
  <img src="https://img.shields.io/badge/Type-Regression-4A90D9?style=flat-square" />
  <img src="https://img.shields.io/badge/Best%20R²-0.6246-2ECC71?style=flat-square" />
  <img src="https://img.shields.io/badge/Model-Gradient%20Boosting-FF6B35?style=flat-square" />
  <img src="https://img.shields.io/badge/Dataset-9%2C551%20records-lightgrey?style=flat-square" />
</p>

---

## Overview

A supervised machine learning regression pipeline that predicts a restaurant's **Aggregate Rating** from its operational and demographic features. Four algorithms — Linear Regression, Decision Tree, Random Forest, and Gradient Boosting — are trained and benchmarked side by side.

---

## Problem Statement

Restaurant platforms need to estimate the expected rating of newly listed restaurants before sufficient user reviews accumulate. Accurate rating prediction helps surface promising restaurants earlier and allows new establishments to benchmark against similar venues.

---

## Objective

Train and evaluate regression models on the Cognifyz restaurant dataset to predict `Aggregate rating` (continuous, 0.0–4.9), identify the best-performing algorithm, and determine which restaurant features carry the most predictive weight.

---

## Technologies Used

| Category | Tool |
|----------|------|
| Language | Python 3.9+ |
| Data manipulation | pandas, numpy |
| Machine learning | scikit-learn |
| Visualization | matplotlib, seaborn |
| Notebook | Jupyter |

---

## Python Libraries

```python
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
```

---

## Dataset Information

| Property | Value |
|----------|-------|
| File | `dataset/Dataset.csv` |
| Total rows | 9,551 |
| Features | 21 columns |
| Target variable | `Aggregate rating` (float, 0.0–4.9) |
| Missing values | 9 (Cuisines column only) |
| Unrated rows removed | 2,148 (rating = 0.0 means "Not Rated") |
| Rows used for training | 7,403 |

---

## Project Workflow

```
Raw Dataset (9,551 rows)
        │
        ▼
1. EDA — shape, dtypes, null counts, target distribution
        │
        ▼
2. Remove unrated rows (rating == 0.0) → 7,403 rows remain
        │
        ▼
3. Fill 9 missing Cuisine values with "Unknown"
        │
        ▼
4. Encode binary columns (Yes/No) with LabelEncoder
        │
        ▼
5. Extract primary cuisine (first item in comma-separated list)
        │
        ▼
6. LabelEncode City and Primary Cuisine
        │
        ▼
7. Drop leakage columns: Rating color, Rating text
   Drop identifier columns: Restaurant ID, Name, Address
        │
        ▼
8. Train/test split: 80% / 20% (random_state=42)
   Train: 5,922 | Test: 1,481
        │
        ▼
9. Train 4 models → evaluate MSE, MAE, R²
        │
        ▼
10. Feature importance analysis (Random Forest)
```

---

## Machine Learning Techniques Used

- **Linear Regression** — baseline model assuming linear feature relationships
- **Decision Tree Regressor** — non-linear, max_depth=8
- **Random Forest Regressor** — ensemble of 100 decision trees (bagging)
- **Gradient Boosting Regressor** — sequential error-correcting ensemble of 100 estimators

**Features selected:**

| Feature | Type | Role |
|---------|------|------|
| Has Table booking | Binary | Service quality signal |
| Has Online delivery | Binary | Convenience offering |
| Price range | Integer | Restaurant tier proxy |
| Average Cost for two | Integer | Actual spend indicator |
| Votes | Integer | Popularity / reach proxy |
| Country Code | Integer | Geographic grouping |
| Cuisine Encoded | Integer | Food category influence |
| City Encoded | Integer | Location market effects |

**Features excluded:**

| Feature | Reason |
|---------|--------|
| Rating color, Rating text | Derived from target — data leakage |
| Restaurant ID, Name, Address | Identifiers, not predictive |
| Currency | Highly correlated with Country Code |

---

## Results

| Model | MSE | MAE | R² |
|-------|-----|-----|-----|
| Linear Regression | 0.1876 | 0.3424 | 0.3933 |
| Decision Tree | 0.1353 | 0.2748 | 0.5624 |
| Random Forest | 0.1256 | 0.2630 | 0.5940 |
| **Gradient Boosting** | **0.1161** | **0.2537** | **0.6246** |

**Best Model: Gradient Boosting (R² = 0.6246)**

The Gradient Boosting model explains ~62% of the variance in restaurant ratings — strong performance given the inherently subjective nature of the target. Top predictors were **Votes**, **Average Cost for two**, and **Price range**, indicating that popularity and price positioning are more influential rating signals than service features like delivery or table booking.

---

## Folder Structure

```
Task-1/
├── README.md
├── notebook.ipynb
├── src/
│   └── rating_predictor.py
├── dataset/
│   └── Dataset.csv
├── outputs/
└── images/
    ├── 01_rating_distribution.png
    ├── 02_model_comparison.png
    ├── 03_actual_vs_predicted.png
    ├── 04_feature_importance.png
    └── 05_votes_vs_rating.png
```

---

## Installation

```bash
cd Task-1
pip install -r ../requirements.txt
```

---

## Usage

**Run the Python script:**

```bash
python src/rating_predictor.py
```

**Run the notebook:**

```bash
jupyter notebook notebook.ipynb
```

Outputs (model comparison plots, actual vs predicted scatter, feature importance chart) are saved to the `outputs/` folder.

---

## Future Improvements

- Hyperparameter tuning with `GridSearchCV` or `Optuna`
- Add interaction features (e.g., `Cost × PriceRange`)
- Incorporate text embeddings on cuisine descriptions
- Explore XGBoost / LightGBM for further R² gains
- Add SHAP values for model explainability

---

## Author

**Abhishek** | Ref: CTI/A1/C358755
BCA — Amity University Online, Noida
Machine Learning Internship @ Cognifyz Technologies
