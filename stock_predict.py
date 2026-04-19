# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║         ADVANCED STOCK PREDICTION & ANALYSIS SYSTEM  v3.0           ║
║                                                                      ║
║  Author   : Spoorthi S Patil                                         ║
║  Version  : 3.0  (Portfolio Edition)                                 ║
║  Purpose  : End-to-end ML pipeline for short-term stock forecasting  ║
║             using ensemble methods, deep learning, technical         ║
║             analysis, backtesting, and explainability tools.         ║
║                                                                      ║
║  MODELS USED:                                                        ║
║    - XGBoost         : Gradient boosting on engineered features      ║
║    - Random Forest   : Ensemble of decision trees (baseline)         ║
║    - BiLSTM          : Bidirectional LSTM for sequential patterns     ║
║                                                                      ║
║  TECHNICAL FEATURES:                                                 ║
║    Open, High, Low, Close, Volume                                    ║
║    MA10, EMA20                        (Trend indicators)             ║
║    RSI                                (Momentum)                     ║
║    MACD, MACD Signal                  (Trend momentum)               ║
║    Bollinger Upper/Lower Band         (Volatility bands)             ║
║    ATR (Average True Range)           (Volatility measure)           ║
║    Daily Return, Log Return           (Price change features)        ║
║                                                                      ║
║  PIPELINE STAGES:                                                    ║
║    1. Data Ingestion & Cleaning                                       ║
║    2. Feature Engineering (15 indicators)                            ║
║    3. Exploratory Data Analysis (EDA)                                ║
║    4. Model Training & Hyperparameter Tuning                         ║
║    5. Evaluation (MAE, RMSE, MAPE, R²)                               ║
║    6. Feature Importance & Explainability                            ║
║    7. Next-Day Prediction with Confidence Scoring                    ║
║    8. 7-Day Rolling Forecast                                         ║
║    9. Backtesting Simulation (Buy/Sell signals & P&L)                ║
║   10. Risk Analysis (Sharpe Ratio, Max Drawdown, Volatility)         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════
#  SECTION 1: IMPORTS & ENVIRONMENT SETUP
# ══════════════════════════════════════════════════════════════════════

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import yfinance as yf

from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, LSTM, Dropout, Input,
                                     Bidirectional, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ── Chart aesthetics ──────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor' : '#0d1117',
    'axes.facecolor'   : '#161b22',
    'axes.edgecolor'   : '#30363d',
    'axes.labelcolor'  : '#c9d1d9',
    'xtick.color'      : '#8b949e',
    'ytick.color'      : '#8b949e',
    'text.color'       : '#c9d1d9',
    'grid.color'       : '#21262d',
    'legend.facecolor' : '#161b22',
    'legend.edgecolor' : '#30363d',
    'font.family'      : 'monospace',
})

ACCENT   = '#58a6ff'   # Blue
ACCENT2  = '#f78166'   # Red/orange
ACCENT3  = '#3fb950'   # Green
ACCENT4  = '#d2a8ff'   # Purple
NEUTRAL  = '#8b949e'   # Grey

# ── Mount Google Drive (for Colab) ────────────────────────────────────
try:
    from google.colab import drive
    drive.mount('/content/drive')
    SAVE_DIR = '/content/drive/MyDrive/StockPredictions/'
    import os
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("[✓] Google Drive mounted. Outputs will be saved to:", SAVE_DIR)
except Exception:
    SAVE_DIR = './'
    print("[i] Running locally. Outputs saved to current directory.")


# ══════════════════════════════════════════════════════════════════════
#  SECTION 2: GLOBAL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

TICKERS = ["MSFT"]


TIME_STEPS  = 20     # Look-back window (trading days)
TRAIN_RATIO = 0.85   # 85 / 15 chronological split
DATA_YEARS  = 5      # Increased to 5 years for richer training data

# All engineered feature columns
FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'MA10', 'EMA20',
    'RSI',
    'MACD', 'MACD_Signal',
    'BB_Upper', 'BB_Lower',
    'ATR',
    'Daily_Return', 'Log_Return'
]

CLOSE_IDX = FEATURE_COLS.index('Close')   # Index 3


# ══════════════════════════════════════════════════════════════════════
#  SECTION 3: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════

def calculate_indicators(df):
    """
    Computes 10 technical indicators on top of raw OHLCV data.

    ┌──────────────────┬──────────────────────────────────────────────┐
    │ Indicator        │ Purpose                                      │
    ├──────────────────┼──────────────────────────────────────────────┤
    │ MA10             │ 10-day Simple Moving Average (short trend)   │
    │ EMA20            │ 20-day Exponential MA (recent-weighted trend)│
    │ RSI (14)         │ Overbought / Oversold momentum (0–100)       │
    │ MACD             │ 12-EMA minus 26-EMA (trend direction)        │
    │ MACD Signal      │ 9-EMA of MACD (crossover triggers)          │
    │ BB Upper/Lower   │ ±2σ Bollinger Bands (volatility envelope)   │
    │ ATR (14)         │ Average True Range (market volatility)       │
    │ Daily Return     │ % price change day-over-day                  │
    │ Log Return       │ log(Pt / Pt-1) — for volatility modelling    │
    └──────────────────┴──────────────────────────────────────────────┘
    """
    close = df['Close']
    high  = df['High']
    low   = df['Low']

    # ── Trend indicators ────────────────────────────────────────────
    df['MA10']  = close.rolling(window=10).mean()
    df['EMA20'] = close.ewm(span=20, adjust=False).mean()

    # ── Momentum: RSI ───────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # ── Trend momentum: MACD ────────────────────────────────────────
    ema12            = close.ewm(span=12, adjust=False).mean()
    ema26            = close.ewm(span=26, adjust=False).mean()
    df['MACD']       = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ── Volatility: Bollinger Bands (20-day, ±2σ) ───────────────────
    ma20            = close.rolling(window=20).mean()
    std20           = close.rolling(window=20).std()
    df['BB_Upper']  = ma20 + 2 * std20
    df['BB_Lower']  = ma20 - 2 * std20

    # ── Volatility: Average True Range (ATR) ────────────────────────
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    # ── Return features ─────────────────────────────────────────────
    df['Daily_Return'] = close.pct_change()
    df['Log_Return']   = np.log(close / close.shift(1))

    return df.dropna()


# ══════════════════════════════════════════════════════════════════════
#  SECTION 4: DATA FETCHING
# ══════════════════════════════════════════════════════════════════════

def get_data(ticker, years=DATA_YEARS):
    """
    Downloads OHLCV data from Yahoo Finance and enriches with features.
    Uses 5 years of history for a larger and more diverse training set.
    """
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = calculate_indicators(df)
    return df[FEATURE_COLS]


# ══════════════════════════════════════════════════════════════════════
#  SECTION 5: EXPLORATORY DATA ANALYSIS (EDA)
# ══════════════════════════════════════════════════════════════════════

def plot_eda(df, ticker):
    """
    Three-panel EDA dashboard:
      Panel 1 — Price history with MA10, EMA20, Bollinger Bands
      Panel 2 — RSI with overbought/oversold zones
      Panel 3 — Feature correlation heatmap

    This section demonstrates understanding of the data before modelling
    — a critical step in any professional ML pipeline.
    """
    print(f"\n[EDA] Generating exploratory analysis for {ticker}...")

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"Exploratory Data Analysis  —  {ticker}", fontsize=16,
                 color='#c9d1d9', y=0.98)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Price + Bands ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df.index, df['Close'],    color=ACCENT,  lw=1.5, label='Close Price')
    ax1.plot(df.index, df['MA10'],     color=ACCENT3, lw=1.0, linestyle='--', label='MA10')
    ax1.plot(df.index, df['EMA20'],    color=ACCENT4, lw=1.0, linestyle='--', label='EMA20')
    ax1.fill_between(df.index, df['BB_Lower'], df['BB_Upper'],
                     alpha=0.12, color=ACCENT2, label='Bollinger Bands')
    ax1.set_title('Price History with Moving Averages & Bollinger Bands', color='#c9d1d9')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: RSI ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df.index, df['RSI'], color=ACCENT2, lw=1.2, label='RSI (14)')
    ax2.axhline(70, color=ACCENT3, linestyle=':', lw=1.2, alpha=0.8)
    ax2.axhline(30, color=ACCENT,  linestyle=':', lw=1.2, alpha=0.8)
    ax2.fill_between(df.index, df['RSI'], 70, where=(df['RSI'] >= 70),
                     alpha=0.15, color=ACCENT3, label='Overbought zone')
    ax2.fill_between(df.index, df['RSI'], 30, where=(df['RSI'] <= 30),
                     alpha=0.15, color=ACCENT,  label='Oversold zone')
    ax2.set_title('RSI — Momentum Oscillator', color='#c9d1d9')
    ax2.set_ylabel('RSI Value')
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: MACD ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df.index, df['MACD'],        color=ACCENT,  lw=1.2, label='MACD')
    ax3.plot(df.index, df['MACD_Signal'], color=ACCENT2, lw=1.0,
             linestyle='--', label='Signal Line')
    macd_hist = df['MACD'] - df['MACD_Signal']
    colors     = [ACCENT3 if v >= 0 else ACCENT2 for v in macd_hist]
    ax3.bar(df.index, macd_hist, color=colors, alpha=0.5, label='Histogram')
    ax3.axhline(0, color=NEUTRAL, lw=0.8, linestyle=':')
    ax3.set_title('MACD — Trend Momentum', color='#c9d1d9')
    ax3.set_ylabel('MACD Value')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Volume ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.bar(df.index, df['Volume'], color=ACCENT4, alpha=0.6, width=1.5)
    vol_ma = df['Volume'].rolling(20).mean()
    ax4.plot(df.index, vol_ma, color=ACCENT2, lw=1.2, label='20-day Vol MA')
    ax4.set_title('Trading Volume', color='#c9d1d9')
    ax4.set_ylabel('Volume')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── Panel 5: Feature Correlation Heatmap ────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    corr = df[['Close', 'MA10', 'EMA20', 'RSI', 'MACD', 'ATR',
               'BB_Upper', 'Daily_Return']].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=ax5, annot=True, fmt='.2f', mask=mask,
                cmap='coolwarm', center=0, linewidths=0.3,
                annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
    ax5.set_title('Feature Correlation Matrix', color='#c9d1d9')
    ax5.tick_params(axis='x', rotation=45, labelsize=8)
    ax5.tick_params(axis='y', rotation=0,  labelsize=8)

    plt.savefig(f"{SAVE_DIR}{ticker}_EDA.png", dpi=150,
                bbox_inches='tight', facecolor='#0d1117')
    plt.show()
    print(f"    [✓] EDA chart saved.")


# ══════════════════════════════════════════════════════════════════════
#  SECTION 6: SEQUENCE BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_sequences(scaled_data, time_steps):
    """
    Converts a 2D scaled array into supervised learning format:
      X → windows of (time_steps × features)
      y → the Close price on the next day (index CLOSE_IDX)

    This 'sliding window' approach lets models learn from temporal
    dependencies — fundamental to time-series ML.
    """
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i - time_steps : i])
        y.append(scaled_data[i, CLOSE_IDX])
    return np.array(X), np.array(y)


# ══════════════════════════════════════════════════════════════════════
#  SECTION 7: MODEL ARCHITECTURES
# ══════════════════════════════════════════════════════════════════════

def build_xgboost(X_train, y_train):
    """
    XGBoost Regressor with tuned hyperparameters.

    Key decisions:
      n_estimators=500  : More trees → lower bias (with early stopping guard)
      max_depth=5       : Moderate depth prevents overfitting
      learning_rate=0.05: Smaller LR + more trees → better generalisation
      subsample=0.8     : Row sampling reduces variance (like dropout)
      colsample_bytree  : Feature sampling — further reduces overfitting
    """
    model = XGBRegressor(
        n_estimators     = 500,
        max_depth        = 5,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 3,
        gamma            = 0.1,
        reg_alpha        = 0.1,      # L1 regularisation
        reg_lambda       = 1.0,      # L2 regularisation
        random_state     = 42,
        n_jobs           = -1,
        verbosity        = 0
    )
    model.fit(X_train, y_train)
    return model


def build_random_forest(X_train, y_train):
    """
    Random Forest baseline — an ensemble of 300 decision trees.

    Included to demonstrate:
      (a) Ensemble understanding beyond gradient boosting
      (b) Comparison baseline to justify the complexity of LSTM/XGBoost
      (c) Feature importance as a secondary explainability check
    """
    model = RandomForestRegressor(
        n_estimators = 300,
        max_depth    = 10,
        max_features = 'sqrt',
        min_samples_split = 5,
        random_state = 42,
        n_jobs       = -1
    )
    model.fit(X_train, y_train)
    return model


def build_lstm(X_train, y_train):
    """
    Bidirectional LSTM architecture with BatchNorm and Dropout.

    Architecture rationale:
      Input           : (TIME_STEPS × n_features)
      BiLSTM(128)     : Forward + backward pass captures both past and
                        future context within the sequence window.
      BatchNorm       : Stabilises gradients, speeds up training.
      Dropout(0.3)    : Regularisation to prevent overfitting.
      BiLSTM(64)      : Second recurrent layer for hierarchical patterns.
      Dense(32)→Dense(1): Final regression head.

    Callbacks:
      EarlyStopping   : Stops when val_loss stops improving (patience=10)
      ReduceLROnPlateau: Halves LR if plateau detected (patience=5)
    """
    n_features = X_train.shape[2]

    model = Sequential([
        Input(shape=(TIME_STEPS, n_features)),
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.25),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer = Adam(learning_rate=1e-3),
        loss      = 'huber'          # Huber loss is more robust to outliers than MSE
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, verbose=0)
    ]

    model.fit(
        X_train, y_train,
        epochs          = 80,
        batch_size      = 32,
        validation_split= 0.1,
        callbacks       = callbacks,
        verbose         = 0
    )
    return model


# ══════════════════════════════════════════════════════════════════════
#  SECTION 8: EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(actual, predicted, label="Model"):
    """
    Computes four regression metrics — a comprehensive evaluation suite:

    ┌───────┬──────────────────────────────────────────────────────────┐
    │ MAE   │ Mean Absolute Error — average USD error per prediction   │
    │ RMSE  │ Root MSE — penalises large errors more heavily           │
    │ MAPE  │ Mean Absolute % Error — scale-independent accuracy       │
    │ R²    │ Coefficient of determination — % variance explained      │
    └───────┴──────────────────────────────────────────────────────────┘

    MAPE < 5%  is considered strong for financial forecasting.
    R² > 0.90  indicates the model captures most price variation.
    """
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((np.array(actual) - np.array(predicted))
                          / (np.array(actual) + 1e-9))) * 100
    r2   = r2_score(actual, predicted)

    print(f"\n  ┌─ {label} Performance ─────────────────────┐")
    print(f"  │  MAE  : ${mae:>8.2f}                         │")
    print(f"  │  RMSE : ${rmse:>8.2f}                         │")
    print(f"  │  MAPE : {mape:>7.2f}%                          │")
    print(f"  │  R²   : {r2:>8.4f}                            │")
    print(f"  └────────────────────────────────────────────┘")

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}


# ══════════════════════════════════════════════════════════════════════
#  SECTION 9: MODEL COMPARISON DASHBOARD
# ══════════════════════════════════════════════════════════════════════

def plot_model_comparison(actual_prices, preds_dict, ticker, metrics_dict):
    """
    Generates a 4-panel model comparison dashboard:
      Panel 1 — Actual vs Predicted for all three models
      Panel 2 — Residuals (prediction errors) over time
      Panel 3 — Metric comparison bar chart (MAE, RMSE, MAPE)
      Panel 4 — Scatter plot: actual vs predicted (best model)
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"Model Performance Comparison  —  {ticker}",
                 fontsize=15, color='#c9d1d9', y=0.98)
    fig.patch.set_facecolor('#0d1117')

    colors = {
        'XGBoost'       : ACCENT,
        'Random Forest' : ACCENT4,
        'BiLSTM'        : ACCENT2
    }

    # ── Panel 1: Actual vs Predicted ────────────────────────────────
    ax = axes[0, 0]
    ax.plot(actual_prices, color='#c9d1d9', lw=2, label='Actual', zorder=5)
    for name, preds in preds_dict.items():
        ax.plot(preds, color=colors[name], lw=1.2,
                linestyle='--', alpha=0.85, label=name)
    ax.set_title('Actual vs Predicted (Test Set)', color='#c9d1d9')
    ax.set_ylabel('Price (USD)')
    ax.set_xlabel('Trading Days')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── Panel 2: Residuals ──────────────────────────────────────────
    ax2 = axes[0, 1]
    for name, preds in preds_dict.items():
        residuals = np.array(actual_prices) - np.array(preds)
        ax2.plot(residuals, color=colors[name], lw=1.0, alpha=0.75, label=name)
    ax2.axhline(0, color='#c9d1d9', lw=1.0, linestyle=':')
    ax2.set_title('Residuals (Actual − Predicted)', color='#c9d1d9')
    ax2.set_ylabel('Error (USD)')
    ax2.set_xlabel('Trading Days')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    # ── Panel 3: Metric Comparison Bar Chart ────────────────────────
    ax3    = axes[1, 0]
    models = list(metrics_dict.keys())
    maes   = [metrics_dict[m]['MAE']  for m in models]
    rmses  = [metrics_dict[m]['RMSE'] for m in models]
    mapes  = [metrics_dict[m]['MAPE'] for m in models]

    x      = np.arange(len(models))
    width  = 0.25
    ax3.bar(x - width, maes,  width, label='MAE',  color=ACCENT,  alpha=0.8)
    ax3.bar(x,         rmses, width, label='RMSE', color=ACCENT4, alpha=0.8)
    ax3.bar(x + width, mapes, width, label='MAPE%',color=ACCENT2, alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, fontsize=9)
    ax3.set_title('Metric Comparison Across Models', color='#c9d1d9')
    ax3.set_ylabel('Error Value')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.25, axis='y')

    # ── Panel 4: Scatter — Best Model ───────────────────────────────
    ax4      = axes[1, 1]
    best_mod = min(metrics_dict, key=lambda m: metrics_dict[m]['MAE'])
    best_p   = preds_dict[best_mod]
    ax4.scatter(actual_prices, best_p, color=ACCENT, alpha=0.4, s=12, label='Predictions')
    lo = min(min(actual_prices), min(best_p))
    hi = max(max(actual_prices), max(best_p))
    ax4.plot([lo, hi], [lo, hi], color=ACCENT3, lw=1.5,
             linestyle='--', label='Perfect fit')
    ax4.set_title(f'Actual vs Predicted Scatter — {best_mod}', color='#c9d1d9')
    ax4.set_xlabel('Actual Price (USD)')
    ax4.set_ylabel('Predicted Price (USD)')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{SAVE_DIR}{ticker}_ModelComparison.png",
                dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.show()
    print(f"    [✓] Model comparison chart saved.")


# ══════════════════════════════════════════════════════════════════════
#  SECTION 10: FEATURE IMPORTANCE & EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════

def plot_feature_importance(xgb_model, rf_model, feature_names, ticker):
    """
    Dual feature importance comparison:
      Left  — XGBoost gain-based importance (how much each feature
               improves the loss when used in a split)
      Right — Random Forest impurity-based importance (mean decrease
               in node impurity across all trees)

    Comparing both gives a more robust picture of which indicators
    genuinely drive the model's predictions — key for interpretability.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Feature Importance Analysis  —  {ticker}",
                 fontsize=14, color='#c9d1d9')
    fig.patch.set_facecolor('#0d1117')

    n = len(feature_names)

    # ── XGBoost ──────────────────────────────────────────────────────
    xgb_imp   = xgb_model.feature_importances_
    xgb_idx   = np.argsort(xgb_imp)
    xgb_names = [feature_names[i] for i in xgb_idx if i < len(feature_names)]
    xgb_vals  = xgb_imp[xgb_idx[:len(xgb_names)]]
    colors_xgb = plt.cm.Blues(np.linspace(0.35, 0.95, n))

    ax1.barh(xgb_names, xgb_vals, color=colors_xgb)
    ax1.set_title('XGBoost — Feature Importance (Gain)', color='#c9d1d9')
    ax1.set_xlabel('Importance Score')
    ax1.grid(True, alpha=0.25, axis='x')
    for i, v in enumerate(xgb_vals):
        ax1.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=8, color='#c9d1d9')

    # ── Random Forest ────────────────────────────────────────────────
    rf_imp   = rf_model.feature_importances_
    rf_idx   = np.argsort(rf_imp)
    rf_names = [feature_names[i] for i in rf_idx if i < len(feature_names)]
    rf_vals  = rf_imp[rf_idx[:len(rf_names)]]
    colors_rf = plt.cm.Purples(np.linspace(0.35, 0.95, n))

    ax2.barh(rf_names, rf_vals, color=colors_rf)
    ax2.set_title('Random Forest — Feature Importance (Impurity)', color='#c9d1d9')
    ax2.set_xlabel('Importance Score')
    ax2.grid(True, alpha=0.25, axis='x')
    for i, v in enumerate(rf_vals):
        ax2.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=8, color='#c9d1d9')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{SAVE_DIR}{ticker}_FeatureImportance.png",
                dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.show()
    print(f"    [✓] Feature importance chart saved.")


# ══════════════════════════════════════════════════════════════════════
#  SECTION 11: INVERSE SCALING HELPER
# ══════════════════════════════════════════════════════════════════════

def inverse_close(scaled_val, scaler, feature_count):
    """
    Reverses MinMax scaling specifically for the Close price column.
    Builds a dummy zero-row, inserts scaled value at CLOSE_IDX,
    then inverts the full transform to recover the original USD price.
    """
    dummy          = np.zeros((1, feature_count))
    dummy[0, CLOSE_IDX] = scaled_val
    return scaler.inverse_transform(dummy)[0, CLOSE_IDX]


# ══════════════════════════════════════════════════════════════════════
#  SECTION 12: PREDICTION HELPERS
# ══════════════════════════════════════════════════════════════════════

def get_analysis_details(mae, current_price, predicted_price, latest_df):
    """
    Produces confidence score, trend label, and multi-indicator reasoning.

    Confidence formula:
        error_pct = MAE / current_price × 100
        confidence = 100 − (error_pct × 10), clamped to [0, 100]

    Extended reasoning now uses: RSI, MACD crossover, Bollinger Band
    position, EMA20, and Volume spike — richer than v2.
    """
    error_pct  = (mae / current_price) * 100
    confidence = max(0, min(100, 100 - (error_pct * 10)))

    pct_change = ((predicted_price - current_price) / current_price) * 100
    if   pct_change >  1.0:  trend = "🟢 Strongly Bullish"
    elif pct_change >  0.3:  trend = "🔵 Mildly Bullish"
    elif pct_change < -1.0:  trend = "🔴 Strongly Bearish"
    elif pct_change < -0.3:  trend = "🟠 Mildly Bearish"
    else:                    trend = "⚪ Neutral (Sideways)"

    # Pull last-row indicator values
    row        = latest_df.iloc[-1]
    rsi        = row['RSI']
    ma10       = row['MA10']
    ema20      = row['EMA20']
    macd       = row['MACD']
    macd_sig   = row['MACD_Signal']
    bb_upper   = row['BB_Upper']
    bb_lower   = row['BB_Lower']
    vol_chg    = latest_df['Volume'].pct_change().iloc[-1]

    reasons = []

    # RSI signal
    if   rsi > 70: reasons.append("RSI overbought (>70) — potential pullback risk")
    elif rsi < 30: reasons.append("RSI oversold (<30) — potential bounce opportunity")
    else:          reasons.append(f"RSI neutral at {rsi:.1f}")

    # MACD crossover
    if   macd > macd_sig: reasons.append("MACD above signal line (bullish crossover)")
    else:                 reasons.append("MACD below signal line (bearish crossover)")

    # Bollinger Band position
    if   current_price >= bb_upper: reasons.append("Price touching upper Bollinger Band")
    elif current_price <= bb_lower: reasons.append("Price touching lower Bollinger Band")

    # EMA20 trend
    if current_price > ema20: reasons.append("Price above EMA20 (upward momentum)")
    else:                     reasons.append("Price below EMA20 (downward pressure)")

    # Volume
    if vol_chg > 0.25: reasons.append("Volume spike detected — high conviction move")

    reason_str = "Signals: " + " | ".join(reasons[:3])
    return round(confidence, 1), trend, reason_str


def predict_next_day(best_model, best_type, latest_scaled, scaler,
                     latest_df, mae_val, ticker):
    """Predicts the next trading day's closing price."""
    curr = latest_df['Close'].iloc[-1].item()

    if best_type == "sequence":
        inp   = latest_scaled[-TIME_STEPS:].reshape(1, TIME_STEPS, len(FEATURE_COLS))
        p_s   = best_model.predict(inp, verbose=0)[0, 0]
    else:
        inp   = latest_scaled[-TIME_STEPS:].reshape(1, -1)
        p_s   = best_model.predict(inp)[0]

    pred = inverse_close(p_s, scaler, len(FEATURE_COLS))
    conf, trend, reason = get_analysis_details(mae_val, curr, pred, latest_df)
    delta = pred - curr
    pct   = (delta / curr) * 100

    print(f"\n{'═'*52}")
    print(f"   NEXT-DAY PREDICTION  —  {ticker}")
    print(f"{'═'*52}")
    print(f"  Current Price  : ${curr:.2f}")
    print(f"  Predicted Price: ${pred:.2f}  ({pct:+.2f}%  {'+' if delta>=0 else ''}{delta:.2f})")
    print(f"  Confidence     : {conf}%")
    print(f"  Trend Signal   : {trend}")
    print(f"  {reason}")
    print(f"{'═'*52}")


def predict_7_day_forecast(best_model, best_type, latest_scaled, scaler,
                            latest_df, mae_val, ticker, actual_prices):
    """
    Rolling 7-day forecast. Each day's prediction becomes the next
    day's input (autoregressive rollout).
    Confidence is reduced by 15% per additional day to reflect
    compounding uncertainty in multi-step forecasts.
    """
    print(f"\n[*] Generating 7-day forecast for {ticker}...")

    curr         = latest_df['Close'].iloc[-1].item()
    rolling_seq  = latest_scaled[-TIME_STEPS:].copy()
    forecast_list = []

    for _ in range(7):
        if best_type == "sequence":
            inp = rolling_seq.reshape(1, TIME_STEPS, len(FEATURE_COLS))
            p_s = best_model.predict(inp, verbose=0)[0, 0]
        else:
            inp = rolling_seq.reshape(1, -1)
            p_s = best_model.predict(inp)[0]

        p_p = inverse_close(p_s, scaler, len(FEATURE_COLS))
        forecast_list.append(p_p)

        new_row          = rolling_seq[-1].copy()
        new_row[CLOSE_IDX] = p_s
        rolling_seq      = np.vstack([rolling_seq[1:], new_row])

    conf, trend, reason = get_analysis_details(
        mae_val, curr, forecast_list[-1], latest_df
    )

    print(f"\n{'═'*52}")
    print(f"   7-DAY FORECAST  —  {ticker}")
    print(f"{'═'*52}")
    print(f"  Trend Signal   : {trend}")
    print(f"  {reason}")
    print(f"{'─'*52}")
    for i, p in enumerate(forecast_list, 1):
        reduced_conf = max(0, conf - 10 * i)
        direction    = "▲" if p > curr else "▼"
        print(f"  Day {i}: ${p:>8.2f}  {direction}  "
              f"({((p - curr)/curr)*100:+.2f}% vs today)   "
              f"[conf: {round(reduced_conf)}%]"
    print(f"{'═'*52}")

    # ── Forecast Chart ────────────────────────────────────────────────
    hist_prices  = actual_prices[-30:]
    hist_idx     = np.arange(len(hist_prices))
    fore_idx     = np.arange(len(hist_prices) - 1, len(hist_prices) + 7)
    fore_vals    = [hist_prices[-1]] + forecast_list

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#0d1117')

    ax.plot(hist_idx, hist_prices, color='#c9d1d9', lw=2, label='Recent History')
    ax.plot(fore_idx, fore_vals,   color=ACCENT2,   lw=2,
            marker='o', markersize=6, label='7-Day Forecast')

    # Uncertainty cone
    for i, (xi, pi) in enumerate(zip(fore_idx[1:], forecast_list), 1):
        spread = mae_val * (1 + 0.15 * i)
        ax.fill_between([xi - 0.4, xi + 0.4],
                        [pi - spread, pi - spread],
                        [pi + spread, pi + spread],
                        alpha=0.2, color=ACCENT2)

    ax.axvline(x=len(hist_prices) - 1, color=NEUTRAL,
               linestyle=':', lw=1.2, label='Today')
    ax.set_title(f"{ticker}  —  7-Day Price Forecast with Uncertainty Bands",
                 fontsize=13, color='#c9d1d9')
    ax.set_ylabel("Price (USD)")
    ax.set_xlabel("Trading Days")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}{ticker}_7DayForecast.png",
                dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.show()
    print(f"    [✓] 7-day forecast chart saved.")


# ══════════════════════════════════════════════════════════════════════
#  SECTION 13: BACKTESTING ENGINE
# ══════════════════════════════════════════════════════════════════════

def run_backtest(actual_prices, predicted_prices, ticker, initial_capital=10000):
    """
    Simulates a simple ML-driven trading strategy on the test set:

    Strategy Rules:
      BUY  — when model predicts price will rise > 0.5% next day
      SELL — when model predicts price will fall > 0.5% next day
      HOLD — otherwise

    Metrics reported:
      Total Return (%)     : Portfolio appreciation
      Sharpe Ratio         : Risk-adjusted return (assumes 0% risk-free rate)
      Max Drawdown (%)     : Largest peak-to-trough decline
      Win Rate (%)         : % of trades that were profitable
      Buy & Hold Return(%) : Baseline passive strategy for comparison

    NOTE: This is a simplified backtest — no transaction costs, slippage,
    or position sizing. Real trading would require a more rigorous setup.
    """
    print(f"\n[BACKTEST] Running simulation for {ticker}...")

    capital   = initial_capital
    shares    = 0
    portfolio = []
    trades    = []
    position  = None

    for i in range(1, len(actual_prices)):
        pred_chg = (predicted_prices[i] - actual_prices[i - 1]) / actual_prices[i - 1]
        price    = actual_prices[i]

        if pred_chg > 0.005 and position != 'long':
            # BUY
            shares   = capital / price
            capital  = 0
            position = 'long'
            trades.append(('BUY', i, price))

        elif pred_chg < -0.005 and position == 'long':
            # SELL
            capital  = shares * price
            shares   = 0
            position = None
            trades.append(('SELL', i, price))

        total_val = capital + shares * price
        portfolio.append(total_val)

    # Close any open position at end
    if shares > 0:
        capital = shares * actual_prices[-1]
        portfolio[-1] = capital

    final_val    = portfolio[-1] if portfolio else initial_capital
    total_return = (final_val - initial_capital) / initial_capital * 100

    # Sharpe Ratio
    daily_rets   = np.diff(portfolio) / np.array(portfolio[:-1])
    sharpe       = (np.mean(daily_rets) / (np.std(daily_rets) + 1e-9)) * np.sqrt(252)

    # Max Drawdown
    port_arr     = np.array(portfolio)
    rolling_max  = np.maximum.accumulate(port_arr)
    drawdowns    = (port_arr - rolling_max) / rolling_max
    max_drawdown = drawdowns.min() * 100

    # Win Rate
    wins         = 0
    buy_price    = None
    for action, idx, price in trades:
        if action == 'BUY':  buy_price = price
        elif action == 'SELL' and buy_price:
            if price > buy_price: wins += 1
    n_complete   = sum(1 for a, _, _ in trades if a == 'SELL')
    win_rate     = (wins / n_complete * 100) if n_complete > 0 else 0

    # Buy-and-Hold baseline
    bh_return    = (actual_prices[-1] - actual_prices[0]) / actual_prices[0] * 100

    print(f"\n  ┌─ Backtest Results — {ticker} ─────────────────┐")
    print(f"  │  Initial Capital    : ${initial_capital:>10,.2f}         │")
    print(f"  │  Final Portfolio    : ${final_val:>10,.2f}         │")
    print(f"  │  Total Return       : {total_return:>+9.2f}%          │")
    print(f"  │  Buy & Hold Return  : {bh_return:>+9.2f}%          │")
    print(f"  │  Sharpe Ratio       : {sharpe:>10.3f}          │")
    print(f"  │  Max Drawdown       : {max_drawdown:>9.2f}%          │")
    print(f"  │  Total Trades       : {len(trades):>10}           │")
    print(f"  │  Win Rate           : {win_rate:>9.1f}%          │")
    print(f"  └──────────────────────────────────────────────┘")

    # ── Backtest Chart ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(f"Backtest Simulation  —  {ticker}", fontsize=14,
                 color='#c9d1d9', y=0.99)
    fig.patch.set_facecolor('#0d1117')

    # Portfolio vs Buy-and-Hold
    bh_portfolio = [initial_capital * (actual_prices[i] / actual_prices[0])
                    for i in range(1, len(actual_prices))]

    ax1 = axes[0]
    ax1.plot(portfolio,    color=ACCENT3, lw=1.8, label='ML Strategy Portfolio')
    ax1.plot(bh_portfolio, color=ACCENT4, lw=1.5,
             linestyle='--', label='Buy & Hold Baseline')
    ax1.axhline(initial_capital, color=NEUTRAL, lw=0.8, linestyle=':')

    for action, idx, price in trades:
        color  = ACCENT3 if action == 'BUY' else ACCENT2
        marker = '^'     if action == 'BUY' else 'v'
        y_val  = portfolio[idx - 1] if idx - 1 < len(portfolio) else portfolio[-1]
        ax1.scatter(idx - 1, y_val, color=color, marker=marker, s=60, zorder=5)

    ax1.set_title('Portfolio Value Over Time', color='#c9d1d9')
    ax1.set_ylabel('Portfolio Value (USD)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    # Drawdown
    ax2 = axes[1]
    ax2.fill_between(range(len(drawdowns)), drawdowns * 100,
                     color=ACCENT2, alpha=0.5, label='Drawdown %')
    ax2.set_title('Drawdown Over Time', color='#c9d1d9')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Trading Days (Test Period)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{SAVE_DIR}{ticker}_Backtest.png",
                dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.show()
    print(f"    [✓] Backtest chart saved.")

    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'bh_return': bh_return
    }


# ══════════════════════════════════════════════════════════════════════
#  SECTION 14: RISK ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def plot_risk_analysis(df, ticker):
    """
    Risk metrics dashboard:
      Panel 1 — Rolling 30-day annualised volatility
      Panel 2 — Daily returns distribution (with normal curve overlay)
      Panel 3 — Value at Risk (VaR 95%) — worst expected 1-day loss
      Panel 4 — Rolling Sharpe Ratio (60-day window)

    These metrics demonstrate quantitative finance knowledge —
    a strong differentiator in ML for finance portfolios.
    """
    print(f"\n[RISK] Generating risk analysis for {ticker}...")

    returns      = df['Daily_Return'].dropna()
    log_ret      = df['Log_Return'].dropna()
    roll_vol     = log_ret.rolling(30).std() * np.sqrt(252) * 100
    roll_sharpe  = (log_ret.rolling(60).mean() / log_ret.rolling(60).std()) * np.sqrt(252)
    var_95       = np.percentile(returns, 5) * 100

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Risk Analysis Dashboard  —  {ticker}",
                 fontsize=14, color='#c9d1d9')
    fig.patch.set_facecolor('#0d1117')

    # Panel 1: Rolling Volatility
    ax1 = axes[0, 0]
    ax1.plot(df.index[len(df) - len(roll_vol):], roll_vol.values,
             color=ACCENT2, lw=1.5)
    ax1.fill_between(df.index[len(df) - len(roll_vol):], roll_vol.values,
                     alpha=0.2, color=ACCENT2)
    ax1.set_title('30-Day Rolling Annualised Volatility (%)', color='#c9d1d9')
    ax1.set_ylabel('Volatility (%)')
    ax1.grid(True, alpha=0.25)

    # Panel 2: Returns Distribution
    ax2 = axes[0, 1]
    ax2.hist(returns * 100, bins=80, color=ACCENT, alpha=0.7,
             edgecolor='none', density=True)
    from scipy.stats import norm
    mu, sigma = returns.mean() * 100, returns.std() * 100
    x_range   = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    ax2.plot(x_range, norm.pdf(x_range, mu, sigma),
             color=ACCENT3, lw=2, label='Normal Curve')
    ax2.axvline(var_95, color=ACCENT2, lw=1.5,
                linestyle='--', label=f'VaR 95%: {var_95:.2f}%')
    ax2.set_title('Daily Returns Distribution', color='#c9d1d9')
    ax2.set_xlabel('Daily Return (%)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    # Panel 3: Cumulative Returns
    ax3 = axes[1, 0]
    cum_ret = (1 + returns).cumprod() - 1
    ax3.plot(df.index[len(df) - len(cum_ret):], cum_ret.values * 100,
             color=ACCENT3, lw=1.5)
    ax3.fill_between(df.index[len(df) - len(cum_ret):],
                     cum_ret.values * 100, 0,
                     where=(cum_ret.values >= 0),
                     alpha=0.2, color=ACCENT3)
    ax3.fill_between(df.index[len(df) - len(cum_ret):],
                     cum_ret.values * 100, 0,
                     where=(cum_ret.values < 0),
                     alpha=0.2, color=ACCENT2)
    ax3.axhline(0, color=NEUTRAL, lw=0.8, linestyle=':')
    ax3.set_title('Cumulative Returns (%)', color='#c9d1d9')
    ax3.set_ylabel('Cumulative Return (%)')
    ax3.grid(True, alpha=0.25)

    # Panel 4: Rolling Sharpe
    ax4 = axes[1, 1]
    ax4.plot(df.index[len(df) - len(roll_sharpe):], roll_sharpe.values,
             color=ACCENT4, lw=1.5)
    ax4.axhline(1.0, color=ACCENT3, lw=0.8, linestyle=':', label='Sharpe = 1 (good)')
    ax4.axhline(0.0, color=NEUTRAL,  lw=0.8, linestyle=':')
    ax4.set_title('60-Day Rolling Sharpe Ratio', color='#c9d1d9')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{SAVE_DIR}{ticker}_RiskAnalysis.png",
                dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.show()

    print(f"  VaR (95%, 1-day): {var_95:.2f}%  "
          f"— On the worst 5% of days, expect ≥ this loss.")
    print(f"    [✓] Risk analysis chart saved.")


# ══════════════════════════════════════════════════════════════════════
#  SECTION 15: MAIN EXECUTION FLOW
# ══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "╔" + "═"*58 + "╗")
    print("║   ADVANCED STOCK PREDICTION & ANALYSIS SYSTEM  v3.0    ║")
    print("║              Portfolio Edition — ML Showcase             ║")
    print("╚" + "═"*58 + "╝")
    print("\n  Available Tickers:")
    print("  " + ", ".join(TICKERS))

    # ── Ticker Selection ──────────────────────────────────────────────
    while True:
        selected_ticker = input("\n  Enter ticker symbol from the list above: ").upper().strip()
        if selected_ticker in TICKERS:
            break
        print(f"  [!] Invalid. Choose from: {', '.join(TICKERS)}")

    # ── Stage 1: Data Ingestion ───────────────────────────────────────
    print(f"\n[1/6] Fetching {DATA_YEARS} years of data for {selected_ticker}...")
    df_data = get_data(selected_ticker)
    print(f"    Shape : {df_data.shape}  ({df_data.shape[0]} trading days × {df_data.shape[1]} features)")
    print(f"    Range : {df_data.index[0].date()}  →  {df_data.index[-1].date()}")
    print(f"    Features: {', '.join(FEATURE_COLS)}")

    # ── Stage 2: EDA ─────────────────────────────────────────────────
    print(f"\n[2/6] Running Exploratory Data Analysis...")
    plot_eda(df_data, selected_ticker)
    plot_risk_analysis(df_data, selected_ticker)

    # ── Stage 3: Preprocessing ────────────────────────────────────────
    scaler      = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_data)
    X, y        = build_sequences(scaled_data, TIME_STEPS)

    split           = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat  = X_test.reshape(X_test.shape[0],  -1)

    print(f"\n  Train samples : {len(X_train)}  |  Test samples : {len(X_test)}")

    # ── Stage 4: Model Training ───────────────────────────────────────
    print(f"\n[3/6] Training three models for {selected_ticker}...")

    print("    → Training XGBoost ...", end=' ', flush=True)
    xgb_model = build_xgboost(X_train_flat, y_train)
    print("Done ✓")

    print("    → Training Random Forest ...", end=' ', flush=True)
    rf_model  = build_random_forest(X_train_flat, y_train)
    print("Done ✓")

    print("    → Training BiLSTM (this may take ~2 min)...")
    lstm_model = build_lstm(X_train, y_train)
    print("    Done ✓")

    # ── Stage 5: Evaluation ───────────────────────────────────────────
    print(f"\n[4/6] Evaluating models on test set...")

    xgb_preds_s   = xgb_model.predict(X_test_flat)
    rf_preds_s    = rf_model.predict(X_test_flat)
    lstm_preds_s  = lstm_model.predict(X_test, verbose=0).flatten()

    actual_prices = [inverse_close(v, scaler, len(FEATURE_COLS)) for v in y_test]
    xgb_prices    = [inverse_close(v, scaler, len(FEATURE_COLS)) for v in xgb_preds_s]
    rf_prices     = [inverse_close(v, scaler, len(FEATURE_COLS)) for v in rf_preds_s]
    lstm_prices   = [inverse_close(v, scaler, len(FEATURE_COLS)) for v in lstm_preds_s]

    metrics = {}
    metrics['XGBoost']       = compute_metrics(actual_prices, xgb_prices,  "XGBoost")
    metrics['Random Forest'] = compute_metrics(actual_prices, rf_prices,   "Random Forest")
    metrics['BiLSTM']        = compute_metrics(actual_prices, lstm_prices,  "BiLSTM")

    # ── Stage 6: Visualisations ───────────────────────────────────────
    print(f"\n[5/6] Generating performance visualisations...")
    preds_dict = {
        'XGBoost'       : xgb_prices,
        'Random Forest' : rf_prices,
        'BiLSTM'        : lstm_prices
    }
    plot_model_comparison(actual_prices, preds_dict, selected_ticker, metrics)
    plot_feature_importance(xgb_model, rf_model,
                        FEATURE_COLS,
                        selected_ticker)

    # ── Select Best Model ─────────────────────────────────────────────
    best_name   = min(metrics, key=lambda m: metrics[m]['MAE'])
    best_model  = {'XGBoost': xgb_model, 'Random Forest': rf_model, 'BiLSTM': lstm_model}[best_name]
    best_type   = "sequence" if best_name == "BiLSTM" else "flat"
    best_preds  = {'XGBoost': xgb_prices, 'Random Forest': rf_prices, 'BiLSTM': lstm_prices}[best_name]
    best_mae    = metrics[best_name]['MAE']

    print(f"\n  ★  Best model: {best_name}  (MAE = ${best_mae:.2f})")

    # ── Backtest ──────────────────────────────────────────────────────
    print(f"\n[6/6] Running backtest simulation...")
    run_backtest(actual_prices, best_preds, selected_ticker)

    # ── Interactive Loop ──────────────────────────────────────────────
    latest_df     = get_data(selected_ticker)
    latest_scaled = scaler.transform(latest_df)

    while True:
        print("\n" + "╔" + "═"*46 + "╗")
        print("║           STOCK COMMAND CENTER                ║")
        print("╠" + "═"*46 + "╣")
        print("║  [P] Next-Day Prediction                      ║")
        print("║  [F] 7-Day Forecast                           ║")
        print("║  [Q] Quit                                     ║")
        print("╚" + "═"*46 + "╝")
        cmd = input("  Enter command: ").lower().strip()

        if cmd == 'q':
            print("\n  Exiting. All charts have been saved to Drive.\n")
            break
        elif cmd == 'p':
            predict_next_day(best_model, best_type, latest_scaled, scaler,
                             latest_df, best_mae, selected_ticker)
        elif cmd == 'f':
            predict_7_day_forecast(best_model, best_type, latest_scaled, scaler,
                                   latest_df, best_mae, selected_ticker, actual_prices)
        else:
            print("  [!] Invalid command. Enter P, F, or Q.")


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
 