import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import LogisticRegressionCV

import os
import xgboost as xgb

MIN_CALIBRATION_TRADES = 100
CALIBRATION_SPLITS = 5


def _build_xgboost_meta_model():
    return XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        scale_pos_weight=0.5,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
    )


def _prepare_xgboost_meta_training_data(X_db, y_db):
    X = X_db.copy().drop(columns=['trade_id'], errors='ignore')
    X = X.select_dtypes(include=['number', 'bool'])
    y = y_db['y_good'].copy().astype(int)

    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    return X, y


class CalibratedXGBMetaLabeler:
    def __init__(self, base_model, feature_names, calibrator=None):
        self.base_model = base_model
        self.feature_names = list(feature_names)
        self.feature_names_in_ = np.array(self.feature_names, dtype=object)
        self.calibrator = calibrator
        self.classes_ = np.array([0, 1], dtype=int)

    def _prepare_features(self, X):
        if isinstance(X, pd.Series):
            X = X.to_frame().T

        prepared = X.copy().drop(columns=['trade_id'], errors='ignore')
        prepared = prepared.select_dtypes(include=['number', 'bool'])
        prepared = prepared.reindex(columns=self.feature_names)

        return prepared.astype(float)

    def predict_proba(self, X):
        prepared = self._prepare_features(X)
        base_probs = self.base_model.predict_proba(prepared)[:, 1]

        if self.calibrator is None:
            final_probs = base_probs
        else:
            final_probs = self.calibrator.predict_proba(base_probs.reshape(-1, 1))[:, 1]

        return np.column_stack([1.0 - final_probs, final_probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ConstantProbabilityMetaLabeler:
    def __init__(self, probability, feature_names):
        self.probability = float(np.clip(probability, 0.0, 1.0))
        self.feature_names = list(feature_names)
        self.feature_names_in_ = np.array(self.feature_names, dtype=object)
        self.classes_ = np.array([0, 1], dtype=int)

    def predict_proba(self, X):
        n_rows = 1 if isinstance(X, pd.Series) else len(X)
        probs = np.full(n_rows, self.probability, dtype=float)
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_xgboost_meta_labeler(X_db, y_db):
    X, y = _prepare_xgboost_meta_training_data(X_db, y_db)

    if X.empty or y.nunique() < 2:
        fallback_prob = 0.5 if y.empty else y.mean()
        return ConstantProbabilityMetaLabeler(fallback_prob, X.columns)

    calibrator = None
    if len(X) >= MIN_CALIBRATION_TRADES:
        oof_probs = pd.Series(np.nan, index=X.index, dtype=float)
        splitter = TimeSeriesSplit(n_splits=CALIBRATION_SPLITS)

        for train_idx, valid_idx in splitter.split(X):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            if y_train.nunique() < 2 or y_valid.nunique() < 2:
                continue

            fold_model = _build_xgboost_meta_model()
            fold_model.fit(X_train, y_train)
            oof_probs.iloc[valid_idx] = fold_model.predict_proba(X_valid)[:, 1]

        valid_calibration_rows = oof_probs.notna()
        if valid_calibration_rows.sum() > 0 and y.loc[valid_calibration_rows].nunique() > 1:
            calibrator = LogisticRegression(random_state=42)
            calibrator.fit(
                oof_probs.loc[valid_calibration_rows].to_numpy().reshape(-1, 1),
                y.loc[valid_calibration_rows].to_numpy(),
            )

    base_model = _build_xgboost_meta_model()
    base_model.fit(X, y)

    return CalibratedXGBMetaLabeler(base_model, X.columns, calibrator=calibrator)




def train_weighted_xgboost(X_db, y_db, feature_cols):
    """
    Trains a robust, shallow XGBoost model weighted by the absolute magnitude of trade returns.
    """
    # Isolate features and target
    X_train = X_db[feature_cols].astype(float)
    y_train = y_db['y_good'].astype(int)
    
    # CALCULATE SAMPLE WEIGHTS
    # Formula: 1.0 + (Absolute Return in Percentage points)
    # A 5% move gets a weight of 6. A 0% move gets a weight of 1.
    weights = 1.0 + (np.abs(y_db['Net_Return']) * 100.0)
    
    # Initialize a Shallow, Anti-Overfit XGBoost Model
    model = xgb.XGBClassifier(
        max_depth=3,             # Shallow trees (avoids memorizing noise)
        learning_rate=0.05,      # Slower, more stable learning
        n_estimators=100,        # Number of trees
        subsample=0.8,           # Randomly drops 20% of rows per tree (fights overfitting)
        colsample_bytree=0.8,    # Randomly drops 20% of features per tree
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train the model with the custom magnitude weights!
    model.fit(X_train, y_train, sample_weight=weights)
    
    return model



#### UNUSED MODELS ######

def train_logistic_meta_labeler(X_db, y_db):
    """
    Trains a self-tuning Logistic Regression model.
    Uses RobustScaler to handle financial outliers and relaxed tolerance for fast convergence.
    """
    # 1. Drop metadata columns
    X = X_db.drop(columns=['trade_id', 'Signal_Time', 'Entry_Time'], errors='ignore')
    y = y_db['y_win']
    
    # 2. Build the Auto-Tuning Pipeline
    model = make_pipeline(
        RobustScaler(),          # <--- TRICK 1: Ignores market crash outliers
        LogisticRegressionCV(use_legacy_attributes=False,
            cv=5,                  
            l1_ratios=[1.0],       
            solver='saga',         
            class_weight='balanced', 
            max_iter=5000,         # Give it slightly more runway
            tol=1e-3,              # <--- TRICK 2: "Close enough" math
            n_jobs=-1,             # Use all CPU cores to speed up the cross-validation
            random_state=42
        )
    )
    
    model.fit(X, y)
    
    # --- DISSERTATION FEATURE IMPORTANCE PRINT ---
    clf = model.named_steps['logisticregressioncv']
    weights = clf.coef_[0]
    killed_features = sum(w == 0 for w in weights)
    print(f"      [L1 Purge] Model deleted {killed_features} out of {len(X.columns)} noisy features.")
    
    return model


from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

def train_random_forest_meta_labeler(X_db, y_db):
    """
    Trains a Calibrated Random Forest for Meta-Labeling.
    Uses strict leaf minimums to prevent overfitting and Platt Scaling 
    to output perfect probabilities for the Kelly Criterion.
    """
    # 1. Drop metadata columns
    X = X_db.drop(columns=['trade_id', 'Signal_Time', 'Entry_Time'], errors='ignore')
    y = y_db['y_win']
    
    # 2. Build the "Stunted" Base Forest
    base_rf = RandomForestClassifier(
        n_estimators=200,                  # More trees = smoother consensus
        max_depth=4,                       # Shallow trees to catch broad market physics
        min_samples_leaf=15,               # <--- TRICK 1: No memorizing individual trades
        max_features='sqrt',               
        class_weight='balanced_subsample', # Better handling of class weights in RFs
        random_state=42, 
        n_jobs=-1              
    )
    
    # 3. Wrap it in a Probability Calibrator
    # cv=5 means it builds 5 separate forests internally to cross-validate the probabilities
    calibrated_rf = CalibratedClassifierCV(
        estimator=base_rf, 
        method='sigmoid',                  # <--- TRICK 2: Platt Scaling for true Kelly probabilities
        cv=5
    )
    
    calibrated_rf.fit(X, y)
    
    return calibrated_rf


if __name__ == "__main__":
    pass
