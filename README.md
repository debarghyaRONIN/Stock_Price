# 📈 Stock Price 

A powerful Streamlit-based web application for stock price analysis, prediction, and options pricing. This app combines technical analysis, machine learning predictions, and financial modeling to provide comprehensive insights into stock market data.

## 🚀 Features

### 1. Stock Data Analysis
- Real-time stock data fetching using yfinance
- Detailed company information and key metrics
- Interactive price charts and historical data visualization
- Downloadable dataset functionality

### 2. Technical Analysis
- Multiple technical indicators:
  - Moving Averages (MA5, MA20, MA50, MA200)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands
- Interactive visualization with Plotly
- Volume analysis and trend indicators

### 3. Price Prediction
- Prophet model integration for time series forecasting
- Customizable prediction parameters:
  - Trend Flexibility
  - Seasonality Strength
  - Holiday Effects
  - Daily/Weekly/Yearly Seasonality
- Confidence interval visualization
- Downloadable prediction results

### 4. Options Pricing Analysis
- Black-Scholes option pricing model
- Interactive option price visualization
- Customizable parameters:
  - Strike Price
  - Time to Maturity
  - Volatility
  - Risk-Free Rate
- Option price heatmaps for different market scenarios

## 🛠️ Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Prophet
- Plotly
- yfinance
- scikit-learn

## 📦 Installation

```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
pip install -r requirements.txt
```

## 🚀 Usage

```bash
streamlit run app.py
```

## 📋 Requirements
- Python 3.7+
- Required packages listed in requirements.txt:
  - streamlit
  - pandas
  - numpy
  - yfinance
  - prophet
  - plotly
  - scikit-learn
  - scipy

## 🎯 Features in Detail

### Stock Information
- Market Cap
- Revenue and Net Income
- EPS and PE Ratios
- Dividend Information
- Trading Volume
- Price Ranges
- Analyst Recommendations

### Technical Analysis
- Multiple timeframe analysis
- Volume-price relationship
- Trend identification
- Momentum indicators
- Interactive charts with zoom capabilities

### Prediction Models
- Prophet model with customizable parameters
- Confidence interval adjustments
- Holiday effect consideration
- Seasonal pattern recognition

### Options Analysis
- Call and Put option pricing
- Interactive price visualization
- Volatility analysis
- Risk assessment tools

## 📊 Data Sources
- Real-time and historical data from Yahoo Finance
- Market holidays and events integration
- Technical indicator calculations

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check [issues page](link-to-issues).

## 📝 License
This project is [MIT](link-to-license) licensed.

## 🙏 Acknowledgments
- Facebook Prophet team for the forecasting library
- Streamlit team for the amazing web framework
- Yahoo Finance for providing financial data
