import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# LOAD DATA
# =====================================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape :", test.shape)

# =====================================================
# FEATURE ENGINEERING
# =====================================================
def create_features(train_df, test_df):
    # Combine to ensure consistent encoding
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    test_df['demand'] = np.nan
    
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # 1. Time Features
    df["hour"] = df["timestamp"].str.split(":").str[0].astype(int)
    df["minute"] = df["timestamp"].str.split(":").str[1].astype(int)
    df["time_of_day"] = df["hour"] * 60 + df["minute"]
    
    # Cyclic time features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
    
    # 2. Target Encoding (Mean demand per Geohash & RoadType based on TRAIN ONLY)
    # This gives the model historical context for specific locations
    geohash_mean = train_df.groupby('geohash')['demand'].mean().to_dict()
    df['geohash_mean_demand'] = df['geohash'].map(geohash_mean)
    
    road_mean = train_df.groupby('RoadType')['demand'].mean().to_dict()
    df['road_mean_demand'] = df['RoadType'].map(road_mean)
    
    weather_mean = train_df.groupby('Weather')['demand'].mean().to_dict()
    df['weather_mean_demand'] = df['Weather'].map(weather_mean)

    # 3. Handle Categoricals
    cat_cols = ["geohash", "RoadType", "LargeVehicles", "Landmarks", "Weather"]
    
    for col in cat_cols:
        # Fill missing values before encoding
        df[col] = df[col].fillna("Unknown")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        # Cast as category for LightGBM's native categorical handling
        df[col] = df[col].astype('category')
        
    # Split back to train and test
    train_processed = df[df['is_train'] == 1].drop(['is_train'], axis=1)
    test_processed = df[df['is_train'] == 0].drop(['is_train', 'demand'], axis=1)
    
    return train_processed, test_processed

print("Engineering features...")
train_feat, test_feat = create_features(train, test)

# =====================================================
# MODEL PREPARATION
# =====================================================
feature_cols = [c for c in train_feat.columns if c not in ["Index", "timestamp", "demand"]]
target_col = "demand"

X = train_feat[feature_cols]
y = train_feat[target_col]
X_test = test_feat[feature_cols]

# =====================================================
# TRAINING WITH KFOLD (Optimized Hyperparameters)
# =====================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
feature_importance = np.zeros(len(feature_cols))

# Advanced tuning parameters for better accuracy
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,      # Slower learning rate for better convergence
    "num_leaves": 63,           # Allows the model to capture more complex patterns
    "max_depth": 8,
    "feature_fraction": 0.8,    # Uses 80% of features per tree to prevent overfitting
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1
}

print("Starting training...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(**lgb_params, n_estimators=5000)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=False),
        ]
    )

    valid_pred = model.predict(X_valid)
    oof_preds[val_idx] = valid_pred
    
    # Accumulate test predictions
    test_preds += model.predict(X_test) / kf.n_splits
    feature_importance += model.feature_importances_ / kf.n_splits

    fold_r2 = r2_score(y_valid, valid_pred)
    print(f"Fold {fold+1} R2 = {fold_r2:.6f}")

# =====================================================
# FINAL SCORING & SUBMISSION
# =====================================================
oof_r2 = r2_score(y, oof_preds)
score = max(0, 100 * oof_r2)

print("\n========================")
print(f"OOF R2 : {oof_r2:.6f}")
print(f"Score  : {score:.4f}")
print("========================")

# Show Top Features
importance_df = pd.DataFrame({"feature": feature_cols, "importance": feature_importance})
importance_df = importance_df.sort_values("importance", ascending=False)
print("\nTop 10 Features:")
print(importance_df.head(10))

# Save submission
test_preds = np.clip(test_preds, 0, None)
submission = pd.DataFrame({
    "Index": test["Index"],
    "demand": test_preds
})

submission.to_csv(
    "submission_optimized.csv",
    index=False
)

print(submission.head())
print("Rows:", len(submission))
print("\nSubmission saved as 'submission_optimized.csv'")