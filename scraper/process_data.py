"""
BitScope | Data Processing Pipeline
====================================
Refactored into an OOP architecture with advanced technical indicators:
RSI, MACD, Bollinger Bands, ATR, and OBV.
"""

import os
import json
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)


# ─────────────────────────────────────────────
# 1. TECHNICAL INDICATOR ENGINE
# ─────────────────────────────────────────────

class TechnicalIndicators:
    """Computes advanced technical indicators on OHLCV data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # --- RSI (Relative Strength Index) ---
    def add_rsi(self, period: int = 14) -> "TechnicalIndicators":
        delta = self.df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        self.df["RSI"] = 100 - (100 / (1 + rs))
        return self

    # --- MACD (Moving Average Convergence Divergence) ---
    def add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> "TechnicalIndicators":
        ema_fast = self.df["Close"].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df["Close"].ewm(span=slow, adjust=False).mean()
        self.df["MACD"] = ema_fast - ema_slow
        self.df["MACD_Signal"] = self.df["MACD"].ewm(span=signal, adjust=False).mean()
        self.df["MACD_Hist"] = self.df["MACD"] - self.df["MACD_Signal"]
        return self

    # --- Bollinger Bands ---
    def add_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> "TechnicalIndicators":
        sma = self.df["Close"].rolling(period).mean()
        std = self.df["Close"].rolling(period).std()
        self.df["BB_Upper"] = sma + std_dev * std
        self.df["BB_Lower"] = sma - std_dev * std
        self.df["BB_Mid"] = sma
        self.df["BB_Width"] = (self.df["BB_Upper"] - self.df["BB_Lower"]) / self.df["BB_Mid"]
        self.df["BB_Pct"] = (self.df["Close"] - self.df["BB_Lower"]) / (
            self.df["BB_Upper"] - self.df["BB_Lower"]
        )
        return self

    # --- ATR (Average True Range) ---
    def add_atr(self, period: int = 14) -> "TechnicalIndicators":
        high_low = self.df["High"] - self.df["Low"]
        high_prev_close = (self.df["High"] - self.df["Close"].shift()).abs()
        low_prev_close = (self.df["Low"] - self.df["Close"].shift()).abs()
        true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        self.df["ATR"] = true_range.ewm(com=period - 1, min_periods=period).mean()
        return self

    # --- OBV (On-Balance Volume) ---
    def add_obv(self) -> "TechnicalIndicators":
        direction = np.sign(self.df["Close"].diff()).fillna(0)
        self.df["OBV"] = (direction * self.df["Volume"]).cumsum()
        return self

    def build(self) -> pd.DataFrame:
        return self.df


# ─────────────────────────────────────────────
# 2. SENTIMENT ENGINE
# ─────────────────────────────────────────────

class SentimentEngine:
    """Loads news JSON and returns a daily averaged sentiment DataFrame."""

    def __init__(self, news_path: str):
        self.news_path = news_path
        self.analyzer = SentimentIntensityAnalyzer()

    def compute(self) -> pd.DataFrame:
        if not os.path.exists(self.news_path):
            print(f"⚠️  News file not found at {self.news_path}. Sentiment will be zero.")
            return pd.DataFrame(columns=["Date", "sentiment", "news_volume"])

        with open(self.news_path, "r") as f:
            news_data = json.load(f)

        records = []
        for item in news_data:
            raw_date = item.get("date", "2026-04-17")
            score = self.analyzer.polarity_scores(item["title"])["compound"]
            records.append({"Date": pd.to_datetime(raw_date).date(), "sentiment": score})

        df = pd.DataFrame(records)
        daily = df.groupby("Date").agg(
            sentiment=("sentiment", "mean"),
            news_volume=("sentiment", "count"),
        ).reset_index()
        return daily


# ─────────────────────────────────────────────
# 3. DATA PIPELINE
# ─────────────────────────────────────────────

class BitScopeDataPipeline:
    """
    End-to-end pipeline:
      Price CSV → Technical Indicators → Sentiment Merge → Target Label → Save
    """

    PRICE_PATH = "../data/raw/btc_price.csv"
    NEWS_PATH = "../data/raw/news.json"
    OUTPUT_PATH = "../data/processed/final_dataset.csv"

    def __init__(self):
        self.price_df: pd.DataFrame = pd.DataFrame()
        self.final_df: pd.DataFrame = pd.DataFrame()

    # --- Step 1: Load Price Data ---
    def _load_price(self) -> "BitScopeDataPipeline":
        df = pd.read_csv(self.PRICE_PATH, skiprows=2)
        df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df.sort_values("Date").reset_index(drop=True)
        self.price_df = df
        print(f"✅ Loaded {len(df)} price rows.")
        return self

    # --- Step 2: Compute Technical Indicators ---
    def _add_indicators(self) -> "BitScopeDataPipeline":
        engine = (
            TechnicalIndicators(self.price_df)
            .add_rsi()
            .add_macd()
            .add_bollinger_bands()
            .add_atr()
            .add_obv()
            .build()
        )
        self.price_df = engine
        print("✅ Technical indicators computed (RSI, MACD, BB, ATR, OBV).")
        return self

    # --- Step 3: Merge Sentiment ---
    def _merge_sentiment(self) -> "BitScopeDataPipeline":
        sentiment_engine = SentimentEngine(self.NEWS_PATH)
        sentiment_df = sentiment_engine.compute()
        merged = pd.merge(self.price_df, sentiment_df, on="Date", how="left")
        merged["sentiment"] = merged["sentiment"].fillna(0)
        merged["news_volume"] = merged["news_volume"].fillna(0)
        self.final_df = merged
        print("✅ Sentiment merged.")
        return self

    # --- Step 4: Create Target & Lag Features ---
    def _engineer_targets(self) -> "BitScopeDataPipeline":
        df = self.final_df
        # Target: 1 if next-day close is higher
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

        # Lag features for temporal context
        for lag in [1, 2, 3]:
            df[f"sentiment_lag{lag}"] = df["sentiment"].shift(lag)
            df[f"return_lag{lag}"] = df["Close"].pct_change(lag)

        # Volatility: rolling 7-day std of returns
        df["volatility_7d"] = df["Close"].pct_change().rolling(7).std()

        self.final_df = df.dropna().reset_index(drop=True)
        print(f"✅ Target and lag features engineered. Final shape: {self.final_df.shape}")
        return self

    # --- Step 5: Save ---
    def _save(self) -> "BitScopeDataPipeline":
        os.makedirs(os.path.dirname(self.OUTPUT_PATH), exist_ok=True)
        self.final_df.to_csv(self.OUTPUT_PATH, index=False)
        print(f"✅ Dataset saved → {self.OUTPUT_PATH}")
        return self

    def run(self) -> pd.DataFrame:
        print("\n🚀 BitScope Data Pipeline Starting...\n" + "─" * 40)
        (
            self._load_price()
            ._add_indicators()
            ._merge_sentiment()
            ._engineer_targets()
            ._save()
        )
        print("─" * 40 + "\n✨ Pipeline complete.\n")
        return self.final_df


# ─────────────────────────────────────────────
# 4. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = BitScopeDataPipeline()
    df = pipeline.run()
    print(df.tail())
