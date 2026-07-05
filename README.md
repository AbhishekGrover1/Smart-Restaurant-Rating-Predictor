# Task-1 · Smart Restaurant Rating Predictor

<p>
  <img src="https://img.shields.io/badge/Type-Supervised%20Regression-4A90D9?style=flat-square" />
  <img src="https://img.shields.io/badge/Best%20R²-0.6246-2ECC71?style=flat-square" />
  <img src="https://img.shields.io/badge/Champion%20Model-Gradient%20Boosting-FF6B35?style=flat-square" />
  <img src="https://img.shields.io/badge/Training%20Set-7%2C403%20records-lightgrey?style=flat-square" />
</p>

---

## Overview

A supervised machine learning regression pipeline that predicts a restaurant's **Aggregate Rating** from its operational and demographic attributes. Four algorithms — Linear Regression, Decision Tree, Random Forest, and Gradient Boosting — are trained, evaluated, and benchmarked on identical train/test splits to identify the best-performing model and the features that carry the most predictive weight.

---

## Problem Statement

Restaurant aggregator platforms need a way to estimate the expected rating of a newly listed venue before meaningful user reviews have accumulated. Reliable rating prediction enables early surfacing of promising restaurants and helps new establishments benchmark their positioning against comparable venues in their market.

---

## Objective

Train and evaluate regression models on the Cognifyz restaurant dataset to predict `Aggregate rating` as a continuous variable (0.0–4.9), determine which algorithm delivers the highest explanatory power, and identify the feature subset that drives predictive performance.

---

## Technologies Used

| Category | Tool |
|----------|------|
| Language | Python 3.9+ |
| Data Manipulation | pandas, numpy |
| Machine Learning | scikit-learn |
| Visualization | matplotlib, seaborn |
| Environment | Jupyter Notebook |

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
| Raw Rows | 9,551 |
| Feature Columns | 21 |
| Target Variable | `Aggregate rating` · float · 0.0–4.9 |
| Missing Values | 9 (Cuisines column only) |
| Unrated Rows Removed | 2,148 · rating = 0.0 indicates "Not Rated" |
| Rows Used for Modelling | 7,403 |
| Train / Test Split | 5,922 · 1,481 (80 / 20 · random_state = 42) |

---

## Project Workflow

```
Raw Dataset  ·  9,551 rows
        │
        ▼
 01  Exploratory Data Analysis
     shape · dtypes · null counts · target distribution
        │
        ▼
 02  Remove unrated rows  (rating == 0.0)
     → 7,403 rows retained
        │
        ▼
 03  Impute 9 missing Cuisine values with "Unknown"
        │
        ▼
 04  Encode binary columns  (Yes / No)  →  LabelEncoder
        │
        ▼
 05  Extract primary cuisine
     (first item in comma-separated cuisine string)
        │
        ▼
 06  LabelEncode  City  and  Primary Cuisine
        │
        ▼
 07  Drop leakage columns  :  Rating color · Rating text
     Drop identifier columns  :  Restaurant ID · Name · Address
        │
        ▼
 08  Train / test split  —  80% / 20%  ·  random_state = 42
        │
        ▼
 09  Train 4 models  →  evaluate MSE · MAE · R²
        │
        ▼
 10  Feature importance analysis  (Random Forest)
```

---

## Machine Learning Techniques Used

| Model | Configuration | Role |
|-------|--------------|------|
| **Linear Regression** | Default sklearn | Baseline — assumes linear feature relationships |
| **Decision Tree Regressor** | max_depth = 8 | Non-linear splits; interpretable structure |
| **Random Forest Regressor** | 100 estimators | Bagging ensemble; reduces overfitting |
| **Gradient Boosting Regressor** | 100 estimators | Sequential error-correction; best generalization |

### Feature Set

**Included:**

| Feature | Data Type | Predictive Role |
|---------|-----------|-----------------|
| Has Table booking | Binary | Service quality signal |
| Has Online delivery | Binary | Operational tier indicator |
| Price range | Integer | Restaurant market positioning |
| Average Cost for two | Integer | Actual spend indicator |
| Votes | Integer | Popularity and review volume proxy |
| Country Code | Integer | Geographic market grouping |
| Cuisine Encoded | Integer | Food category influence |
| City Encoded | Integer | Local market dynamics |

**Excluded:**

| Feature | Exclusion Reason |
|---------|-----------------|
| Rating color · Rating text | Derived directly from the target — data leakage |
| Restaurant ID · Name · Address | Non-predictive identifiers |
| Currency | Highly collinear with Country Code |

---

## Results

| Model | MSE ↓ | MAE ↓ | R² ↑ |
|-------|--------|--------|------|
| Linear Regression | 0.1876 | 0.3424 | 0.3933 |
| Decision Tree | 0.1353 | 0.2748 | 0.5624 |
| Random Forest | 0.1256 | 0.2630 | 0.5940 |
| **Gradient Boosting** | **0.1161** | **0.2537** | **0.6246** |

**Champion Model: Gradient Boosting · R² = 0.6246**

The Gradient Boosting regressor explains approximately 62% of the variance in restaurant ratings — strong performance given the inherently subjective nature of the target. Top predictive features were **Votes**, **Average Cost for two**, and **Price range**, indicating that popularity signals and price positioning carry substantially more weight than operational features such as delivery availability or table booking.

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

**Execute the script:**

```bash
python src/rating_predictor.py
```

**Launch the notebook:**

```bash
jupyter notebook notebook.ipynb
```

All outputs — model comparison charts, actual vs. predicted scatter plots, and the feature importance bar chart — are written to the `outputs/` directory.

---

## Future Improvements

- Hyperparameter optimisation with `GridSearchCV` or `Optuna`
- Engineered interaction features — e.g. `Cost × PriceRange`
- Text embeddings on cuisine descriptions for richer feature representation
- Evaluation of XGBoost and LightGBM for additional R² gains
- SHAP value integration for production-grade model explainability

---

## Author

**Abhishek** · Ref: CTI/A1/C358755
BCA · Amity University Online, Noida
Machine Learning Internship · Cognifyz Technologies

---
---
