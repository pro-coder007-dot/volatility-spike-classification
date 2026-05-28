# 📈 Volatility Spike Predictor (NVDA Stock ML System)

A machine learning system that predicts **volatility spikes in NVIDIA (NVDA) stock prices** using historical OHLCV data. The project combines **financial time-series analysis, feature engineering, and an API-based deployment using FastAPI**.

---

## 🚀 Project Overview

This project focuses on predicting whether a stock will experience a **volatility spike** (sudden large price movement) rather than predicting exact prices.

Instead of forecasting price, the model answers:

> “Will the stock become highly volatile soon?”

---

## 🎯 Problem Statement

Stock prices often behave unpredictably due to market activity, trading volume, and external factors. This project aims to:

- Detect periods of **high volatility**
- Classify market conditions as:
  - 🟢 Stable (no spike)
  - 🔴 Volatile spike

---

## 📊 Dataset

The dataset contains historical stock data for NVIDIA (NVDA):

### Features:
- Date
- Open
- High
- Low
- Close
- Adj Close
- Volume

### Engineered Features:
- Daily Returns
- Rolling Volatility (5–10 day window)
- Spike Label (Target Variable)

---

## 🧠 Target Variable

We define:

- **Spike (1)** → High volatility period (top 10% volatility values)
- **No Spike (0)** → Normal market behavior

---

## ⚙️ Tech Stack

- Python 🐍
- Pandas / NumPy
- Scikit-learn
- FastAPI 🚀
- Uvicorn
- Matplotlib / Seaborn (for analysis)

---

## 🏗️ Project Pipeline

```text
1. Load NVDA dataset
2. Clean and preprocess data
3. Feature engineering (returns, volatility)
4. Create spike labels
5. Train ML model
6. Evaluate performance
7. Deploy using FastAPI