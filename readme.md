# Stock Price Prediction — ML + Deep Learning

> End-to-end stock forecasting pipeline that fetches live data, engineers 15 technical indicators, trains and compares three ML models, runs a backtesting simulation, and generates next-day and 7-day price forecasts with confidence scores.

---

## Results on MSFT

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| **Random Forest** | **$13.46** | **$16.66** | **2.85%** | **0.895** |
| XGBoost | $25.98 | $32.24 | 5.23% | 0.607 |
| BiLSTM | $49.73 | $57.66 | 10.30% | -0.257 |

**Random Forest won** — an important finding. On structured financial tabular data with engineered features, ensemble tree methods can outperform deep learning. The BiLSTM's negative R² indicates it struggled to generalise on this test period.

**Backtest on MSFT:**
- ML Strategy: −13.5% · Buy & Hold: −17.0% → model preserved more capital
- Sharpe Ratio: −0.879 · Max Drawdown: −23.67% · Win Rate: 50%

---

## Charts

### EDA — Price History, RSI, MACD, Volume, Correlation Heatmap
![EDA](MSFT_EDA.png)

### Risk Analysis — Volatility, Returns Distribution, Cumulative Returns, Sharpe
![Risk Analysis](MSFT_RiskAnalysis.png)

### Model Comparison — Actual vs Predicted, Residuals, Metrics, Scatter
![Model Comparison](MSFT_ModelComparison.png)

### Feature Importance — XGBoost (Gain) vs Random Forest (Impurity)
![Feature Importance](MSFT_FeatureImportance.png)

### Backtest — Portfolio vs Buy & Hold, Drawdown
![Backtest](MSFT_Backtest.png)

### 7-Day Forecast — Price Forecast with Uncertainty Bands
![7 Day Forecast](MSFT_7DayForecast.png)

---

## What This Project Does

Runs a complete ML pipeline for any of 24 supported tickers:

1. **Fetches 5 years of live OHLCV data** from Yahoo Finance
2. **Engineers 15 features** — raw price data plus 10 technical indicators
3. **Runs EDA** — price history with Bollinger Bands, RSI, MACD, volume, correlation heatmap
4. **Runs risk analysis** — rolling volatility, returns distribution, VaR (95%), rolling Sharpe ratio
5. **Trains 3 models** — XGBoost, Random Forest, and a Bidirectional LSTM
6. **Evaluates and compares** all three on a held-out 15% test set using MAE, RMSE, MAPE, and R²
7. **Plots feature importance** for XGBoost (gain-based) and Random Forest (impurity-based)
8. **Runs a backtest** — simulates a buy/sell strategy driven by the best model's predictions
9. **Predicts next-day price** with a confidence score and multi-indicator signal reasoning
10. **Generates a 7-day rolling forecast** with compounding uncertainty bands

---

## Quick Start

```bash
# Requires Python 3.11 (TensorFlow does not support Python 3.12+)
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install numpy pandas matplotlib seaborn yfinance scikit-learn xgboost tensorflow scipy

python stock_predict.py
```

You'll be prompted to pick a ticker:

```
Available Tickers:
TSLA, MSFT, PG, META, AMZN, GOOG, AMD, AAPL, NFLX, TSM,
KO, F, COST, DIS, VZ, CRM, INTC, BA, NOC, PYPL, ENPH, NIO, ZS, XPEV

Enter ticker symbol from the list above: MSFT
```

After training (~2–3 min), an interactive menu appears:

```
[P] Next-Day Prediction
[F] 7-Day Forecast
[Q] Quit
```

---

## Technical Features Engineered

| Feature | Description |
|---|---|
| MA10 | 10-day Simple Moving Average — short-term trend |
| EMA20 | 20-day Exponential MA — recent-weighted trend |
| RSI (14) | Momentum oscillator, 0–100 — overbought/oversold signal |
| MACD | 12-EMA minus 26-EMA — trend direction |
| MACD Signal | 9-EMA of MACD — crossover trigger |
| BB Upper/Lower | ±2σ Bollinger Bands — volatility envelope |
| ATR (14) | Average True Range — daily volatility measure |
| Daily Return | % price change day over day |
| Log Return | log(Pt / Pt−1) — for volatility modelling |

---

## Model Architectures

**XGBoost** — 500 trees, max depth 5, learning rate 0.05, L1+L2 regularisation, row and column subsampling. Flat input: 20 days × 15 features = 300-dimensional vector.

**Random Forest** — 300 trees, max depth 10, sqrt feature sampling. Same flat input as XGBoost. Included as an interpretable baseline.

**Bidirectional LSTM** — BiLSTM(128) → BatchNorm → Dropout(0.3) → BiLSTM(64) → BatchNorm → Dropout(0.25) → Dense(32) → Dense(1). Huber loss, Adam optimiser, EarlyStopping (patience=10), ReduceLROnPlateau (patience=5). Sequential input: 20 × 15 tensor.

---

## Sample Prediction Output

```
════════════════════════════════════════════════════
   NEXT-DAY PREDICTION  —  MSFT
════════════════════════════════════════════════════
  Current Price  : $422.79
  Predicted Price: $410.55  (-2.89%  -12.24)
  Confidence     : 68.2%
  Trend Signal   : 🔴 Strongly Bearish
  Signals: RSI overbought (>70) — potential pullback risk
           | MACD above signal line (bullish crossover)
           | Price touching upper Bollinger Band
════════════════════════════════════════════════════
```

---

## Why Random Forest Beat LSTM

Worth understanding, not hiding. The BiLSTM underperformed because:

- A 20-day window doesn't expose strong enough sequential dependencies to justify added model complexity
- Engineered features (RSI, MACD, ATR) already capture temporal patterns explicitly, reducing the advantage of recurrent architectures
- The test period was bearish for MSFT — deep learning models are more sensitive to distribution shift

This is an honest evaluation, which is the point of comparing all three.

---

## Supported Tickers

```
TSLA  MSFT  PG    META  AMZN  GOOG  AMD   AAPL
NFLX  TSM   KO    F     COST  DIS   VZ    CRM
INTC  BA    NOC   PYPL  ENPH  NIO   ZS    XPEV
```

---

## Stack

`Python 3.11` `TensorFlow / Keras` `XGBoost` `scikit-learn` `yfinance` `pandas` `numpy` `matplotlib` `seaborn` `scipy`
