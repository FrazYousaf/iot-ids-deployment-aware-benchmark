import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

import config
from utils import ensure_dirs, set_all_seeds, save_json

set_all_seeds(config.RANDOM_SEED)
ensure_dirs(config.ARTIFACT_DIR, config.RESULTS_DIR, config.MODELS_DIR)

print("Loading dataset:", config.DATASET_PATH)
df = pd.read_csv(config.DATASET_PATH)

print("Original shape:", df.shape)
print("\nOriginal label distribution:")
print(df[config.LABEL_COLUMN].value_counts())

# Basic cleaning
df = df.copy()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=[config.LABEL_COLUMN], inplace=True)

# Keep numeric feature columns only
feature_cols = [c for c in df.columns if c != config.LABEL_COLUMN]
numeric_feature_cols = []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(df[c]):
        numeric_feature_cols.append(c)

dropped_non_numeric = sorted(set(feature_cols) - set(numeric_feature_cols))
if dropped_non_numeric:
    print("\nDropped non-numeric columns:", dropped_non_numeric)

X_df = df[numeric_feature_cols].copy()
y_raw = df[config.LABEL_COLUMN].astype(str).copy()

# Drop rows where all feature values are missing
all_missing_mask = X_df.isna().all(axis=1)
if all_missing_mask.sum() > 0:
    X_df = X_df.loc[~all_missing_mask]
    y_raw = y_raw.loc[~all_missing_mask]

print("\nFeature shape before optional correlation filter:", X_df.shape)

# Optional high-correlation filter
if config.DROP_HIGH_CORR_FEATURES:
    print("\nApplying high-correlation feature filter...")
    corr = X_df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > config.CORR_THRESHOLD)]
    X_df.drop(columns=to_drop, inplace=True)
    print("Dropped highly correlated features:", to_drop)

feature_names_full = X_df.columns.tolist()
print("Final full feature count:", len(feature_names_full))

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
class_names = label_encoder.classes_.tolist()

print("\nClasses:", class_names)
print("\nLabel distribution after cleaning:")
print(pd.Series(y_raw).value_counts())

# Train/val/test split: first test, then validation from train pool
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_df,
    y,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_SEED,
    stratify=y,
)

relative_val_size = config.VAL_SIZE / (1.0 - config.TEST_SIZE)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=relative_val_size,
    random_state=config.RANDOM_SEED,
    stratify=y_trainval,
)

print("\nSplit shapes:")
print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)

# Impute using train only
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train).astype(np.float32)
X_val_imp = imputer.transform(X_val).astype(np.float32)
X_test_imp = imputer.transform(X_test).astype(np.float32)

# Scaler for neural networks using train only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp).astype(np.float32)
X_val_scaled = scaler.transform(X_val_imp).astype(np.float32)
X_test_scaled = scaler.transform(X_test_imp).astype(np.float32)

# Save preprocessors
joblib.dump(label_encoder, os.path.join(config.ARTIFACT_DIR, "label_encoder.joblib"))
joblib.dump(imputer, os.path.join(config.ARTIFACT_DIR, "imputer.joblib"))
joblib.dump(scaler, os.path.join(config.ARTIFACT_DIR, "scaler.joblib"))

# Feature selection using XGBoost on train set only
print("\nTraining XGBoost feature selector...")
xgb_params = dict(
    n_estimators=250,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=config.RANDOM_SEED,
    n_jobs=-1,
    tree_method="hist",
)

feature_selector = XGBClassifier(**xgb_params)

start = time.perf_counter()
try:
    if config.USE_GPU_FOR_XGBOOST:
        feature_selector.set_params(device="cuda")
    feature_selector.fit(X_train_imp, y_train)
except Exception as e:
    print("GPU XGBoost failed or unavailable. Falling back to CPU.")
    print("Reason:", str(e))
    feature_selector = XGBClassifier(**xgb_params)
    feature_selector.fit(X_train_imp, y_train)
end = time.perf_counter()

print("Feature selector training time sec:", round(end - start, 3))

importances = feature_selector.feature_importances_
importance_df = pd.DataFrame({
    "feature": feature_names_full,
    "importance": importances,
}).sort_values("importance", ascending=False)

importance_path = os.path.join(config.ARTIFACT_DIR, "feature_importance_xgboost.csv")
importance_df.to_csv(importance_path, index=False)

top_features = importance_df.head(config.TOP_K_FEATURES)["feature"].tolist()
top_indices = [feature_names_full.index(f) for f in top_features]

print("\nTop selected features:")
print(top_features)

save_json(
    {
        "top_k": config.TOP_K_FEATURES,
        "selected_features": top_features,
        "full_features": feature_names_full,
        "classes": class_names,
    },
    os.path.join(config.ARTIFACT_DIR, "feature_selection.json"),
)

# Save prepared arrays
def save_npz(name, Xtr, Xv, Xte, feature_names):
    path = os.path.join(config.ARTIFACT_DIR, name)
    np.savez_compressed(
        path,
        X_train=Xtr,
        X_val=Xv,
        X_test=Xte,
        y_train=y_train.astype(np.int64),
        y_val=y_val.astype(np.int64),
        y_test=y_test.astype(np.int64),
        class_names=np.array(class_names, dtype=object),
        feature_names=np.array(feature_names, dtype=object),
    )
    print("Saved:", path)

save_npz("full_tree.npz", X_train_imp, X_val_imp, X_test_imp, feature_names_full)
save_npz("full_scaled.npz", X_train_scaled, X_val_scaled, X_test_scaled, feature_names_full)

save_npz(
    "reduced_tree.npz",
    X_train_imp[:, top_indices],
    X_val_imp[:, top_indices],
    X_test_imp[:, top_indices],
    top_features,
)

save_npz(
    "reduced_scaled.npz",
    X_train_scaled[:, top_indices],
    X_val_scaled[:, top_indices],
    X_test_scaled[:, top_indices],
    top_features,
)

print("\nPreparation complete.")
print("Artifacts saved in:", config.ARTIFACT_DIR)
