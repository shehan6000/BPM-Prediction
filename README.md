"""
Kaggle Playground Series - BPM Prediction
Complete solution with EDA, feature engineering, and ensemble modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("="*80)
print("LOADING DATA")
print("="*80)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nColumns: {train.columns.tolist()}")

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)

# Basic statistics
print("\nTarget variable statistics:")
print(train['BeatsPerMinute'].describe())

# Check for missing values
print(f"\nMissing values in train:\n{train.isnull().sum().sum()}")
print(f"Missing values in test:\n{test.isnull().sum().sum()}")

# Check data types
print(f"\nData types:\n{train.dtypes.value_counts()}")

# Identify feature columns (exclude id and target)
if 'id' in train.columns:
    feature_cols = [col for col in train.columns if col not in ['id', 'BeatsPerMinute']]
else:
    feature_cols = [col for col in train.columns if col != 'BeatsPerMinute']

print(f"\nNumber of features: {len(feature_cols)}")
print(f"Feature columns: {feature_cols[:10]}...")  # Show first 10

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

def create_features(df, is_train=True):
    """Create additional features from existing ones"""
    df = df.copy()
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if 'id' in numeric_cols:
        numeric_cols = [col for col in numeric_cols if col not in ['id', 'BeatsPerMinute']]
    
    # Statistical features
    df['feature_mean'] = df[numeric_cols].mean(axis=1)
    df['feature_std'] = df[numeric_cols].std(axis=1)
    df['feature_max'] = df[numeric_cols].max(axis=1)
    df['feature_min'] = df[numeric_cols].min(axis=1)
    df['feature_range'] = df['feature_max'] - df['feature_min']
    df['feature_median'] = df[numeric_cols].median(axis=1)
    
    # Interaction features (top correlations if available)
    if len(numeric_cols) >= 2:
        df['feature_prod_01'] = df[numeric_cols[0]] * df[numeric_cols[1]]
        df['feature_sum_01'] = df[numeric_cols[0]] + df[numeric_cols[1]]
    
    if len(numeric_cols) >= 3:
        df['feature_prod_02'] = df[numeric_cols[0]] * df[numeric_cols[2]]
    
    return df

# Apply feature engineering
print("Creating features for train...")
train_fe = create_features(train, is_train=True)
print("Creating features for test...")
test_fe = create_features(test, is_train=False)

print(f"New train shape: {train_fe.shape}")
print(f"New test shape: {test_fe.shape}")

# ============================================================================
# 4. PREPARE DATA FOR MODELING
# ============================================================================
print("\n" + "="*80)
print("PREPARING DATA FOR MODELING")
print("="*80)

# Separate features and target
X = train_fe.drop(['BeatsPerMinute'], axis=1)
if 'id' in X.columns:
    X = X.drop(['id'], axis=1)
y = train_fe['BeatsPerMinute']

X_test = test_fe.copy()
if 'id' in X_test.columns:
    test_ids = X_test['id'].copy()
    X_test = X_test.drop(['id'], axis=1)
else:
    test_ids = test['id'].copy() if 'id' in test.columns else np.arange(len(test))

# Ensure train and test have same columns
X_test = X_test[X.columns]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X_test shape: {X_test.shape}")

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 5. BASELINE MODEL - SIMPLE MEAN
# ============================================================================
print("\n" + "="*80)
print("BASELINE MODEL")
print("="*80)

baseline_pred = np.full(len(y), y.mean())
baseline_rmse = np.sqrt(mean_squared_error(y, baseline_pred))
print(f"Baseline (Mean) RMSE: {baseline_rmse:.4f}")
print(f"Mean BPM: {y.mean():.4f}")
print(f"Std BPM: {y.std():.4f}")

# ============================================================================
# 6. MODEL TRAINING WITH CROSS-VALIDATION
# ============================================================================
print("\n" + "="*80)
print("MODEL TRAINING")
print("="*80)

# Setup cross-validation
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

# Dictionary to store results
models = {}
cv_scores = {}
predictions = {}

# Ridge Regression
print("\n--- Ridge Regression ---")
ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
ridge_scores = -cross_val_score(ridge, X_scaled, y, cv=kf, 
                                 scoring='neg_root_mean_squared_error', n_jobs=-1)
cv_scores['Ridge'] = ridge_scores
print(f"CV RMSE: {ridge_scores.mean():.4f} (+/- {ridge_scores.std():.4f})")

# Train on full data and predict
ridge.fit(X_scaled, y)
models['Ridge'] = ridge
predictions['Ridge'] = ridge.predict(X_test_scaled)

# Lasso Regression
print("\n--- Lasso Regression ---")
lasso = Lasso(alpha=0.1, random_state=RANDOM_STATE)
lasso_scores = -cross_val_score(lasso, X_scaled, y, cv=kf, 
                                 scoring='neg_root_mean_squared_error', n_jobs=-1)
cv_scores['Lasso'] = lasso_scores
print(f"CV RMSE: {lasso_scores.mean():.4f} (+/- {lasso_scores.std():.4f})")

lasso.fit(X_scaled, y)
models['Lasso'] = lasso
predictions['Lasso'] = lasso.predict(X_test_scaled)

# ElasticNet
print("\n--- ElasticNet ---")
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE)
elastic_scores = -cross_val_score(elastic, X_scaled, y, cv=kf, 
                                   scoring='neg_root_mean_squared_error', n_jobs=-1)
cv_scores['ElasticNet'] = elastic_scores
print(f"CV RMSE: {elastic_scores.mean():.4f} (+/- {elastic_scores.std():.4f})")

elastic.fit(X_scaled, y)
models['ElasticNet'] = elastic
predictions['ElasticNet'] = elastic.predict(X_test_scaled)

# Random Forest
print("\n--- Random Forest ---")
rf = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=10,
                           min_samples_leaf=4, random_state=RANDOM_STATE, n_jobs=-1)
rf_scores = -cross_val_score(rf, X, y, cv=kf, 
                              scoring='neg_root_mean_squared_error', n_jobs=-1)
cv_scores['RandomForest'] = rf_scores
print(f"CV RMSE: {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")

rf.fit(X, y)
models['RandomForest'] = rf
predictions['RandomForest'] = rf.predict(X_test)

# Gradient Boosting
print("\n--- Gradient Boosting ---")
gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                               random_state=RANDOM_STATE)
gb_scores = -cross_val_score(gb, X, y, cv=kf, 
                              scoring='neg_root_mean_squared_error', n_jobs=-1)
cv_scores['GradientBoosting'] = gb_scores
print(f"CV RMSE: {gb_scores.mean():.4f} (+/- {gb_scores.std():.4f})")

gb.fit(X, y)
models['GradientBoosting'] = gb
predictions['GradientBoosting'] = gb.predict(X_test)

# ============================================================================
# 7. ENSEMBLE - WEIGHTED AVERAGE
# ============================================================================
print("\n" + "="*80)
print("ENSEMBLE MODEL")
print("="*80)

# Calculate weights based on inverse CV RMSE (better models get more weight)
cv_means = {name: scores.mean() for name, scores in cv_scores.items()}
best_rmse = min(cv_means.values())

# Inverse weighting
weights = {}
for name, rmse in cv_means.items():
    weights[name] = 1.0 / rmse

# Normalize weights
total_weight = sum(weights.values())
weights = {name: w / total_weight for name, w in weights.items()}

print("\nModel weights:")
for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:20s}: {weight:.4f} (CV RMSE: {cv_means[name]:.4f})")

# Create ensemble prediction
ensemble_pred = np.zeros(len(X_test))
for name, pred in predictions.items():
    ensemble_pred += weights[name] * pred

predictions['Ensemble'] = ensemble_pred

# ============================================================================
# 8. CREATE SUBMISSION FILE
# ============================================================================
print("\n" + "="*80)
print("CREATING SUBMISSION")
print("="*80)

# Use the best performing model or ensemble
submission = pd.DataFrame({
    'id': test_ids,
    'BeatsPerMinute': ensemble_pred
})

submission.to_csv('submission_ensemble.csv', index=False)
print("Submission file saved: submission_ensemble.csv")
print(f"\nSubmission statistics:")
print(submission['BeatsPerMinute'].describe())

# Save individual model predictions as well
for name, pred in predictions.items():
    if name != 'Ensemble':
        sub = pd.DataFrame({
            'id': test_ids,
            'BeatsPerMinute': pred
        })
        filename = f'submission_{name.lower()}.csv'
        sub.to_csv(filename, index=False)
        print(f"Saved: {filename}")

# ============================================================================
# 9. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\nCross-validation RMSE scores:")
for name in sorted(cv_means.items(), key=lambda x: x[1]):
    print(f"{name[0]:20s}: {name[1]:.4f}")

print(f"\nBaseline RMSE: {baseline_rmse:.4f}")
print(f"Best model improvement: {(baseline_rmse - min(cv_means.values())) / baseline_rmse * 100:.2f}%")

print("\n" + "="*80)
print("DONE! Check your submission files.")
print("="*80)
