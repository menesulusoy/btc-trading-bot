# BTC/USDT Machine Learning Decision Support and Paper Trading System

A machine learning-based Bitcoin market direction prediction, backtesting, and real-time paper trading dashboard.

This project analyzes hourly **BTC/USDT** market data, generates technical indicators, predicts whether the next hourly closing price will move upward, and converts the model output into simulated trading decisions using predefined risk-management rules.

The final prediction model combines **Random Forest**, **XGBoost**, and **LightGBM** through a **soft-voting ensemble**. Model development and evaluation are performed in a Jupyter Notebook, while the trained model is loaded by a Flask application for live monitoring and paper trading.

> [!WARNING]
> This project is intended for educational, research, and paper trading purposes only. It does not execute real trades and does not constitute financial or investment advice. Historical backtest results do not guarantee future performance.

---

## Overview

The system consists of two main stages:

1. **Model training and backtesting**
   - Downloads historical hourly BTC/USDT OHLCV data from Binance using `ccxt`
   - Calculates technical indicators and engineered features
   - Trains a hybrid classification model
   - Evaluates the model on chronologically separated test data
   - Runs a paper trading backtest with risk-management rules
   - Saves the trained model and test accuracy to `btc_hybrid_model.pkl`

2. **Live dashboard and paper trading**
   - Loads the previously trained model without retraining it
   - Fetches recent hourly BTC/USDT market data
   - Recalculates the same model features
   - Produces an upward-movement probability score
   - Generates simulated BUY, SELL, STOP-LOSS, TAKE-PROFIT, or HOLD signals
   - Updates the dashboard approximately every five seconds through Socket.IO

---

## Main Features

- Hourly BTC/USDT direction prediction
- Historical data collection from the Binance API
- Chronological train/test split for time-series integrity
- Hybrid soft-voting classification model
- Random Forest, XGBoost, and LightGBM integration
- Technical indicator-based feature engineering
- Historical backtesting
- Configurable confidence thresholds
- Stop-loss and take-profit rules
- Trading commission simulation
- Minimum waiting period between selected decisions
- Strong-confidence BUY exception
- Real-time Flask and Socket.IO dashboard
- Simulated BTC and USDT portfolio tracking
- Live unrealized and realized profit/loss display
- Trading history and bot signal monitoring
- Buy-and-hold performance comparison

---

## Technology Stack

### Machine Learning and Data Processing

- Python
- Jupyter Notebook
- pandas
- NumPy
- scikit-learn
- XGBoost
- LightGBM
- `ccxt`

### Web Application

- Flask
- Flask-SocketIO
- HTML
- CSS
- JavaScript

### Market Data

- Binance API
- BTC/USDT hourly OHLCV candles

---

## Machine Learning Problem

The project treats short-term Bitcoin direction prediction as a binary classification problem.

The target variable is defined as:

```text
1 -> The next hourly close is higher than the current close
0 -> The next hourly close is not higher than the current close
```

The objective is not to predict the exact future price. Instead, the model estimates the probability that the next hourly closing price will move upward.

---

## Dataset

| Property | Value |
|---|---|
| Data source | Binance API |
| Trading pair | BTC/USDT |
| Timeframe | 1-hour candles |
| Date range | 2023-06-03 00:00 to 2026-06-02 23:00 |
| Raw observations | 26,304 |
| Train/test split | 80% training, 20% testing |
| Clean training observations | 20,992 |
| Clean test observations | 5,211 |
| Split method | Chronological, without shuffling |

The dataset uses the standard OHLCV structure:

- Open
- High
- Low
- Close
- Volume

Because this is financial time-series data, the records are not randomly shuffled. Earlier observations are used for training, and later observations are used for testing to reduce the risk of data leakage.

---

## Model Architecture

The final model is a `VotingClassifier` that combines three algorithms through soft voting:

| Model | Purpose |
|---|---|
| Random Forest | Captures nonlinear patterns across different market conditions |
| XGBoost | Improves direction classification through gradient boosting |
| LightGBM | Adds an efficient and complementary boosting approach |
| Soft Voting | Combines class probabilities from all three models |

The final prediction is based on the average class probabilities produced by the ensemble members.

The trained model is saved as:

```text
btc_hybrid_model.pkl
```

The saved object also includes the model test accuracy used by the dashboard.

The Flask application does **not** retrain the model. It loads the existing `.pkl` file and uses it directly for live prediction.

---

## Model Features

The final model uses the following 11 features:

| Feature | Description |
|---|---|
| `SMA_20` | 20-period simple moving average |
| `SMA_50` | 50-period simple moving average |
| `RSI` | Relative Strength Index |
| `MACD` | Moving Average Convergence Divergence |
| `MACD_Hist` | MACD histogram |
| `BB_Width` | Bollinger Band width |
| `BB_Pos` | Price position within the Bollinger Bands |
| `ATR_14` | 14-period Average True Range |
| `Hacim_Ratio` | Current volume divided by the 20-period average volume |
| `Getiri_%` | One-hour percentage return |
| `Getiri_3h` | Three-hour short-term return |

The dashboard also displays **Stochastic %K** for market monitoring. However, Stochastic %K is not included in the final model feature list.

> The feature names in the source code may remain in Turkish because the trained model expects the same column names used during training.

---

## Model Performance

The final model achieved the following results on the chronological test set:

| Metric | Result |
|---|---:|
| Training accuracy | 56.28% |
| Test accuracy | 52.91% |
| Class 0 precision | 0.53 |
| Class 0 recall | 0.50 |
| Class 0 F1-score | 0.51 |
| Class 1 precision | 0.53 |
| Class 1 recall | 0.56 |
| Class 1 F1-score | 0.54 |
| Test support | 5,211 observations |

A test accuracy of approximately 52.91% is only moderately above random classification. This is expected in noisy and highly volatile financial time series and should not be interpreted as a guarantee of profitable future trading.

For this reason, the project evaluates the system with both machine learning metrics and strategy-level metrics such as:

- Backtest return
- Win rate
- Number of trades
- Commission impact
- Stop-loss behavior
- Take-profit behavior
- Buy-and-hold comparison

---

## Trading Strategy

The raw model prediction is not converted directly into a trade. It is filtered through a set of strategy and risk-management rules.

### Final Strategy Parameters

| Parameter | Value |
|---|---:|
| Initial balance | 10,000 USDT |
| Position size | 80% |
| Normal confidence threshold | 0.58 |
| Strong confidence threshold | 0.70 |
| Stop-loss | 1.5% |
| Take-profit | 3.0% |
| Minimum waiting period | 6 hourly candles |
| Commission | 0.01% |

### BUY Logic

A normal BUY signal may be generated when:

```text
upward_probability >= 0.58
AND no position is currently open
AND the minimum waiting period has been completed
```

A strong BUY signal may be generated when:

```text
upward_probability >= 0.70
AND no position is currently open
```

The strong-confidence rule can bypass the normal waiting period to reduce the chance of missing a high-confidence opportunity.

### SELL Logic

An open position may be closed when one of the following conditions is met:

- The stop-loss threshold is reached
- The take-profit threshold is reached
- The model's upward score falls below the required threshold
- The strategy's normal exit conditions are satisfied

The system can generate the following signal types:

```text
AL
SAT
SAT-SL
SAT-TP
HOLD
```

These labels correspond to BUY, SELL, SELL-STOP-LOSS, SELL-TAKE-PROFIT, and HOLD.

---

## Backtest Results

Using the selected strategy parameters and an initial balance of 10,000 USDT, the example backtest produced:

| Metric | Result |
|---|---:|
| Final portfolio value | 12,662.19 USDT |
| Net profit | 2,662.19 USDT |
| Bot return | +26.62% |
| Buy-and-hold return | -41.15% |
| Win rate | 59.8% |

These results apply only to the specific historical test period and selected parameters. They should not be interpreted as expected future returns.

---

## Dashboard

The live dashboard displays:

- Current BTC/USDT price
- 24-hour price change
- Candlestick chart
- Total portfolio value
- USDT balance
- BTC balance
- Win rate
- Total trade count
- Model test accuracy
- RSI
- Model upward probability
- Stochastic %K
- ATR volatility
- Bot signals
- Trade history
- Realized profit/loss
- Unrealized profit/loss for an open position

The Flask application retrieves the latest 120 hourly candles, recalculates the model features, and computes the upward score using:

```python
model.predict_proba(...)
```

Socket.IO sends dashboard updates approximately every five seconds without requiring a full page refresh.

Because the current hourly candle may still be open, live signals should be treated as simulation and monitoring outputs rather than confirmed market-close predictions.

---

## Application Workflow

```text
Binance API
    |
    v
Hourly BTC/USDT OHLCV Data
    |
    v
Data Cleaning and Feature Engineering
    |
    v
Random Forest + XGBoost + LightGBM
    |
    v
Soft-Voting Ensemble
    |
    +----------------------------+
    |                            |
    v                            v
Model Evaluation             Backtesting
    |                            |
    +-------------+--------------+
                  |
                  v
      btc_hybrid_model.pkl
                  |
                  v
             Flask app1.py
                  |
                  v
        Live Paper Trading Dashboard
```

---

## Recommended Repository Structure

The exact structure may differ depending on the current repository files, but a clean organization could look like this:

```text
btc-trading-bot/
├── app1.py
├── btc_hybrid_model.pkl
├── notebooks/
│   └── model_training_and_backtest.ipynb
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   └── js/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/menesulusoy/btc-trading-bot.git
cd btc-trading-bot
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

When a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

A typical installation may include:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm ccxt flask flask-socketio
```

Additional packages may be required depending on the exact notebook and dashboard implementation.

---

## Running the Project

### Train and Save the Model

Open the model training notebook and run the cells in order.

The notebook should:

1. Download historical BTC/USDT data
2. Calculate the technical indicators
3. Create the target variable
4. Split the dataset chronologically
5. Train the ensemble model
6. Calculate performance metrics
7. Run the backtest
8. Save the trained model as `btc_hybrid_model.pkl`

### Start the Dashboard

Make sure `btc_hybrid_model.pkl` is located where `app1.py` expects it, then run:

```bash
python app1.py
```

Open the local address shown in the terminal, typically:

```text
http://127.0.0.1:5000
```

The exact host and port may differ depending on the Flask configuration.

---

## Configuration

The main strategy parameters are typically defined in `app1.py` or the notebook:

```python
CONFIDENCE = 0.58
STRONG_CONFIDENCE = 0.70
STOP_LOSS = 0.015
TAKE_PROFIT = 0.030
POZISYON_ORAN = 0.80
MIN_BEKLE = 6
KOMISYON = 0.0001
```

Changing these values can significantly affect:

- Number of trades
- Risk exposure
- Drawdown
- Win rate
- Portfolio return
- Comparison with buy-and-hold

Parameters should be evaluated on unseen data rather than selected only for the best historical result.

---

## Limitations

- The model predicts direction, not the exact future price.
- Test accuracy is only moderately above random chance.
- Cryptocurrency markets are highly volatile and non-stationary.
- Backtest performance can be affected by parameter selection.
- The live dashboard may evaluate an incomplete hourly candle.
- Exchange latency, slippage, spread, outages, and liquidity are not fully represented.
- Historical relationships may not remain stable in future market conditions.
- A profitable historical backtest does not guarantee profitable live performance.
- The system is designed for paper trading and decision support, not automated real-money execution.

---

## Possible Improvements

- Walk-forward validation
- Time-series cross-validation
- Automated hyperparameter optimization
- Probability calibration
- Feature importance analysis
- SHAP-based explainability
- Dynamic position sizing
- Trailing stop-loss
- Maximum drawdown monitoring
- Sharpe and Sortino ratio reporting
- More realistic slippage and spread simulation
- Multiple cryptocurrency support
- Database-backed trade history
- CSV and Excel report export
- User-configurable strategy settings
- Automatic model retraining
- Docker support
- Cloud deployment
- Authentication for the dashboard
- Closed-candle-only signal generation
- Out-of-sample forward testing

---

## Disclaimer

This software is provided for educational and research purposes only.

It is not financial advice, investment advice, or a recommendation to buy or sell any asset. Cryptocurrency trading involves substantial risk, including the possible loss of all invested capital.

The authors and contributors are not responsible for any financial losses, damages, or decisions made using this project.

---

## Author

**Murat Enes Ulusoy**

Artificial Intelligence Project  
Machine Learning-Based Cryptocurrency Decision Support and Paper Trading System

---

## License

No license is currently specified.

Before reusing, distributing, or modifying this project, add an appropriate license file such as MIT, Apache-2.0, or another license that matches the intended usage.
