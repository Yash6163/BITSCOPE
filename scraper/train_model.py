"""
BitScope | Model Training Pipeline
====================================
OOP-based trainer with GridSearchCV for XGBoost hyperparameter tuning.
Optimized for MacBook Air (n_jobs=-1 uses all cores efficiently).
"""

import os
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


class BitScopeTrainer:
    """
    Trains an XGBoost classifier with time-series-aware cross-validation
    and grid-search hyperparameter tuning.
    """

    DATA_PATH = "../data/processed/final_dataset.csv"
    MODEL_PATH = "../models/bitscope_xgb.pkl"
    SCALER_PATH = "../models/bitscope_scaler.pkl"

    FEATURES = [
        "Close", "Volume", "sentiment", "news_volume",
        "RSI", "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Width", "BB_Pct", "ATR", "OBV",
        "sentiment_lag1", "sentiment_lag2", "sentiment_lag3",
        "return_lag1", "return_lag2", "return_lag3",
        "volatility_7d",
    ]

    # ── Hyper-parameter search space ──────────────────────────────
    # Kept small for MacBook Air (no GPU). Expand on cloud if needed.
    PARAM_GRID = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 1.0],
        "reg_alpha": [0, 0.1],          # L1 regularization → prevents overfitting
        "reg_lambda": [1, 1.5],         # L2 regularization → stabilizes weights
    }

    def __init__(self):
        self.df = pd.DataFrame()
        self.model = None
        self.scaler = StandardScaler()

    def _load_data(self):
        if not os.path.exists(self.DATA_PATH):
            raise FileNotFoundError(f"❌ {self.DATA_PATH} not found. Run process_data.py first.")
        self.df = pd.read_csv(self.DATA_PATH).dropna(subset=self.FEATURES + ["Target"])
        print(f"✅ Loaded data: {self.df.shape[0]} rows, {len(self.FEATURES)} features.")

    def _split(self):
        X = self.df[self.FEATURES]
        y = self.df["Target"]
        # 80/20 temporal split — no shuffle to preserve time ordering
        cut = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:cut], X.iloc[cut:]
        y_train, y_test = y.iloc[:cut], y.iloc[cut:]

        # Scale features (important for OBV & Volume which dwarf others)
        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc = self.scaler.transform(X_test)
        return X_train_sc, X_test_sc, y_train, y_test

    def _tune_and_train(self, X_train, y_train):
        """
        TimeSeriesSplit ensures each CV fold respects temporal order,
        preventing future data from leaking into training windows.
        """
        tscv = TimeSeriesSplit(n_splits=5)
        base_model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,          # all CPU cores (efficient on M-series chips)
            tree_method="hist", # fast histogram method
        )

        print("\n🔍 Running GridSearchCV (this may take ~2–5 minutes)...\n")
        grid = GridSearchCV(
            estimator=base_model,
            param_grid=self.PARAM_GRID,
            scoring="accuracy",
            cv=tscv,
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)

        print(f"\n🏆 Best params: {grid.best_params_}")
        print(f"🏆 Best CV accuracy: {grid.best_score_:.4f}")
        self.model = grid.best_estimator_

    def _evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"\n📊 Test Accuracy: {acc * 100:.2f}%")
        print("\n" + classification_report(y_test, preds, target_names=["DOWN ↓", "UP ↑"]))
        return acc

    def _save(self):
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        joblib.dump(self.model, self.MODEL_PATH)
        joblib.dump(self.scaler, self.SCALER_PATH)
        print(f"✅ Model saved → {self.MODEL_PATH}")
        print(f"✅ Scaler saved → {self.SCALER_PATH}")

    def run(self):
        print("\n🚀 BitScope Model Trainer\n" + "─" * 40)
        self._load_data()
        X_train, X_test, y_train, y_test = self._split()
        self._tune_and_train(X_train, y_train)
        self._evaluate(X_test, y_test)
        self._save()
        print("─" * 40 + "\n✨ Training complete.\n")


if __name__ == "__main__":
    trainer = BitScopeTrainer()
    trainer.run()
