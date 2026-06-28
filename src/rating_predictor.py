# ============================================================
# Task 1: Smart Restaurant Rating Predictor
# Intern: Abhishek | Enrollment: CTI/A1/C358755
# Organization: Cognifyz Technologies
# Course: BCA | Domain: Machine Learning
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

# ── Output folder for screenshots ─────────────────────────
SCREENSHOTS = os.path.join(os.path.dirname(__file__), '..', 'Screenshots')
os.makedirs(SCREENSHOTS, exist_ok=True)

# ── 1. Load Dataset ────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'Dataset.csv'))
print(f"Dataset loaded. Shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

# ── 2. Exploratory Analysis ────────────────────────────────
print("=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

print("\nBasic Info:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nTarget variable (Aggregate rating) distribution:")
print(df['Aggregate rating'].describe())

# Ratings with 0.0 are 'Not rated' restaurants → we exclude them
# because 0.0 is not a real rating, it would bias any model
not_rated = (df['Aggregate rating'] == 0).sum()
print(f"\nRestaurants with 0.0 rating (Not Rated): {not_rated}")
df = df[df['Aggregate rating'] > 0].copy()
print(f"Dataset after removing unrated rows: {df.shape}")

# ── 3. Handle Missing Values ───────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Handling Missing Values")
print("=" * 60)

# Only 'Cuisines' has 9 missing rows; fill with 'Unknown'
df['Cuisines'] = df['Cuisines'].fillna('Unknown')
print("Filled missing Cuisines with 'Unknown'")

# ── 4. Feature Engineering ─────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Feature Selection & Encoding")
print("=" * 60)

# Features chosen based on actual dataset columns:
# - Has Table booking   → binary service feature
# - Has Online delivery → binary service feature
# - Price range         → numeric, proxy for restaurant tier
# - Average Cost for two → numeric, actual spend
# - Votes               → numeric, proxy for popularity
# - Country Code        → numeric identifier for geography
# Excluded: Restaurant ID, Name, Address (identifiers not predictive)
# Excluded: Rating color, Rating text (derived from target → data leakage)
# Excluded: Longitude/Latitude (handled in Task 4)
# Excluded: Currency (correlated with Country Code)

le = LabelEncoder()

df['Has Table booking'] = le.fit_transform(df['Has Table booking'])
df['Has Online delivery'] = le.fit_transform(df['Has Online delivery'])
df['Is delivering now'] = le.fit_transform(df['Is delivering now'])

# Extract primary cuisine (first listed) and encode
df['Primary Cuisine'] = df['Cuisines'].apply(lambda x: x.split(',')[0].strip())
df['Cuisine Encoded'] = le.fit_transform(df['Primary Cuisine'])

# City encoding
df['City Encoded'] = le.fit_transform(df['City'])

FEATURES = [
    'Has Table booking',
    'Has Online delivery',
    'Price range',
    'Average Cost for two',
    'Votes',
    'Country Code',
    'Cuisine Encoded',
    'City Encoded',
]

TARGET = 'Aggregate rating'

X = df[FEATURES]
y = df[TARGET]

print(f"Features selected: {FEATURES}")
print(f"Target: {TARGET}")
print(f"Feature matrix shape: {X.shape}")

# ── 5. Train / Test Split ──────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Train-Test Split (80/20)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ── 6. Train Multiple Models ───────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Training Models")
print("=" * 60)

models = {
    'Linear Regression':        LinearRegression(),
    'Decision Tree':            DecisionTreeRegressor(max_depth=8, random_state=42),
    'Random Forest':            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting':        GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse  = mean_squared_error(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    results[name] = {'MSE': round(mse, 4), 'MAE': round(mae, 4), 'R2': round(r2, 4), 'model': model, 'preds': preds}
    print(f"\n{name}:")
    print(f"  MSE : {mse:.4f}")
    print(f"  MAE : {mae:.4f}")
    print(f"  R²  : {r2:.4f}")

# ── 7. Select Best Model ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Best Model Selection")
print("=" * 60)

best_name = max(results, key=lambda k: results[k]['R2'])
best      = results[best_name]
print(f"Best Model: {best_name}")
print(f"  R² = {best['R2']}, MAE = {best['MAE']}, MSE = {best['MSE']}")

# ── 8. Visualizations ─────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Generating Visualizations")
print("=" * 60)

sns.set_theme(style='whitegrid', palette='muted')

# 8a. Rating distribution (after removing unrated)
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df['Aggregate rating'], bins=20, color='steelblue', edgecolor='white')
ax.set_title('Distribution of Restaurant Ratings', fontsize=14, fontweight='bold')
ax.set_xlabel('Aggregate Rating')
ax.set_ylabel('Count')
fig.tight_layout()
fig.savefig(os.path.join(SCREENSHOTS, '01_rating_distribution.png'), dpi=150)
plt.close(fig)
print("Saved: 01_rating_distribution.png")

# 8b. Model comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics = ['MSE', 'MAE', 'R2']
colors  = ['#e07b54', '#5b8db8', '#6bbf84']
for ax, metric, color in zip(axes, metrics, colors):
    names  = list(results.keys())
    values = [results[n][metric] for n in names]
    bars = ax.bar(names, values, color=color, edgecolor='white', width=0.5)
    ax.set_title(metric, fontsize=13, fontweight='bold')
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                str(val), ha='center', va='bottom', fontsize=8)
fig.suptitle('Model Performance Comparison', fontsize=15, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(SCREENSHOTS, '02_model_comparison.png'), dpi=150)
plt.close(fig)
print("Saved: 02_model_comparison.png")

# 8c. Actual vs Predicted scatter (best model)
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, best['preds'], alpha=0.3, color='steelblue', edgecolors='none', s=20)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_title(f'Actual vs Predicted — {best_name}', fontsize=13, fontweight='bold')
ax.set_xlabel('Actual Rating')
ax.set_ylabel('Predicted Rating')
fig.tight_layout()
fig.savefig(os.path.join(SCREENSHOTS, '03_actual_vs_predicted.png'), dpi=150)
plt.close(fig)
print("Saved: 03_actual_vs_predicted.png")

# 8d. Feature importance (Random Forest only)
rf_model = results['Random Forest']['model']
importances = rf_model.feature_importances_
feat_series = pd.Series(importances, index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
feat_series.plot(kind='barh', ax=ax, color='teal', edgecolor='white')
ax.set_title('Feature Importance — Random Forest', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
fig.tight_layout()
fig.savefig(os.path.join(SCREENSHOTS, '04_feature_importance.png'), dpi=150)
plt.close(fig)
print("Saved: 04_feature_importance.png")

# 8e. Votes vs Rating scatter
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df['Votes'], df['Aggregate rating'], alpha=0.15, color='purple', s=10)
ax.set_title('Votes vs Aggregate Rating', fontsize=13, fontweight='bold')
ax.set_xlabel('Number of Votes')
ax.set_ylabel('Aggregate Rating')
fig.tight_layout()
fig.savefig(os.path.join(SCREENSHOTS, '05_votes_vs_rating.png'), dpi=150)
plt.close(fig)
print("Saved: 05_votes_vs_rating.png")

# ── 9. Summary ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"\nBest Model  : {best_name}")
print(f"MSE         : {best['MSE']}")
print(f"MAE         : {best['MAE']}")
print(f"R² Score    : {best['R2']}")
print("\nConclusion:")
print("  Votes and Price range are the strongest predictors of")
print("  aggregate rating. Gradient Boosting / Random Forest")
print("  outperform linear approaches due to non-linear patterns.")
print("\nAll visualizations saved to Screenshots folder.")
