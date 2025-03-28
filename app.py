import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import warnings
import scipy.stats as si

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide",
)

# App title and description
st.title("📈 Stock Price Prediction")
st.write("This app fetches historical stock data, predicts future prices, and displays interactive visualizations.")

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

# Stock ticker input
ticker = st.sidebar.text_input("Stock Ticker Symbol", "AAPL")

# Date range selection
today = datetime.now()
start_date = st.sidebar.date_input("Start Date", today - timedelta(days=365*2))
end_date = st.sidebar.date_input("End Date", today)

# Prediction days
prediction_days = st.sidebar.slider("Prediction Days", 7, 90, 30)

# Add Prophet model parameters to sidebar (after prediction_days)
st.sidebar.subheader("Prophet Model Parameters")
changepoint_prior_scale = st.sidebar.slider("Trend Flexibility", 0.001, 0.5, 0.05, 0.001)
seasonality_prior_scale = st.sidebar.slider("Seasonality Strength", 0.01, 10.0, 10.0, 0.1)
holidays_prior_scale = st.sidebar.slider("Holiday Effect", 0.01, 10.0, 10.0, 0.1)
daily_seasonality = st.sidebar.checkbox("Daily Seasonality", value=True)
weekly_seasonality = st.sidebar.checkbox("Weekly Seasonality", value=True)
yearly_seasonality = st.sidebar.checkbox("Yearly Seasonality", value=True)
confidence_interval = st.sidebar.slider("Confidence Interval", 0.8, 0.99, 0.95, 0.01)

# Add option pricing parameters
st.sidebar.header("Option Pricing Parameters")
current_price = st.sidebar.number_input("Current Asset Price", min_value=1.0, value=100.0, step=1.0)
strike_price = st.sidebar.number_input("Strike Price", min_value=1.0, value=100.0, step=1.0)
time_to_maturity = st.sidebar.slider("Time to Maturity (years)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
volatility = st.sidebar.slider("Volatility (%)", min_value=1, max_value=100, value=20, step=1) / 100
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", min_value=0.1, max_value=10.0, value=5.0, step=0.1) / 100

# Heatmap parameters
st.sidebar.header("Heatmap Parameters")
min_spot = st.sidebar.number_input("Min Spot Price", min_value=1.0, value=current_price * 0.7, step=1.0)
max_spot = st.sidebar.number_input("Max Spot Price", min_value=min_spot + 10, value=current_price * 1.3, step=1.0)
min_vol = st.sidebar.slider("Min Volatility (%)", min_value=1, max_value=90, value=10, step=1) / 100
max_vol = st.sidebar.slider("Max Volatility (%)", min_value=int(min_vol * 100) + 5, max_value=100, value=50, step=1) / 100

@st.cache_data
def load_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            st.error(f"No data found for ticker {ticker}")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def format_large_number(num):
    """Format large numbers in billions (B) or trillions (T)"""
    if num is None or np.isnan(num):
        return "N/A"
    
    if num >= 1_000_000_000_000:  # Trillion
        return f"${num / 1_000_000_000_000:.2f}T"
    elif num >= 1_000_000_000:  # Billion
        return f"${num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:  # Million
        return f"${num / 1_000_000:.2f}M"
    else:
        return f"${num:,.2f}"

def get_detailed_stock_info(ticker_symbol):
    """Get detailed stock information and format it properly"""
    try:
        ticker_data = yf.Ticker(ticker_symbol)
        info = ticker_data.info
        
        # Create a dictionary with all the metrics we want to display
        stock_metrics = {
            "Current Price": f"${info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))}",
            "Market Cap": format_large_number(info.get('marketCap')),
            "Revenue (ttm)": format_large_number(info.get('totalRevenue')),
            "Net Income (ttm)": format_large_number(info.get('netIncomeToCommon')),
            "Shares Out": f"{info.get('sharesOutstanding', 0) / 1_000_000_000:.2f}B" if info.get('sharesOutstanding') else "N/A",
            "EPS (ttm)": f"${info.get('trailingEps', 'N/A')}",
            "PE Ratio": f"{info.get('trailingPE', 'N/A')}",
            "Forward PE": f"{info.get('forwardPE', 'N/A')}",
            "Dividend": f"${info.get('dividendRate', 0):.2f} ({info.get('dividendYield', 0) * 100:.2f}%)" if info.get('dividendRate') else "N/A",
            "Ex-Dividend Date": info.get('exDividendDate', 'N/A'),
            "Volume": f"{info.get('volume', 0):,}",
            "Open": f"${info.get('open', 'N/A')}",
            "Previous Close": f"${info.get('regularMarketPreviousClose', 'N/A')}",
            "Day's Range": f"${info.get('dayLow', 0):.2f} - ${info.get('dayHigh', 0):.2f}",
            "52-Week Range": f"${info.get('fiftyTwoWeekLow', 0):.2f} - ${info.get('fiftyTwoWeekHigh', 0):.2f}",
            "Beta": f"{info.get('beta', 'N/A')}",
            "Analysts": info.get('recommendationKey', 'N/A').capitalize(),
            "Price Target": f"${info.get('targetMeanPrice', 0):.2f} ({((info.get('targetMeanPrice', 0) / info.get('currentPrice', 1)) - 1) * 100:.2f}%)"
        }
        
        return stock_metrics
    except Exception as e:
        st.warning(f"Could not load complete stock information: {e}")
        return {}

def plot_technical_indicators(data):
    # Calculate RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Plot
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='purple'))
    )
    
    # Add RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    
    fig.update_layout(
        height=600,
        title_text=f"RSI Technical Indicator for {ticker}",
        showlegend=True,
        yaxis=dict(
            title="RSI Value",
            range=[0, 100]
        ),
        xaxis_title="Date"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def predict_with_prophet(data, periods):
    # Prepare data for Prophet
    try:
        df_prophet = data.reset_index()[['Date', 'Close']]
        df_prophet.columns = ['ds', 'y']
        
        # Initialize and fit the model with custom parameters
        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            interval_width=confidence_interval
        )
        
        # Add US holidays
        model.add_country_holidays(country_name='US')
        
        # Fit the model
        model.fit(df_prophet)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods)
        
        # Make prediction
        forecast = model.predict(future)
        
        # Plot forecast
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=df_prophet['ds'],
            y=df_prophet['y'],
            mode='lines',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast['ds'][-periods:],
            y=forecast['yhat'][-periods:],
            mode='lines',
            name='Forecast',
            line=dict(color='red')
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast['ds'][-periods:], forecast['ds'][-periods:].iloc[::-1]]),
            y=pd.concat([forecast['yhat_upper'][-periods:], forecast['yhat_lower'][-periods:].iloc[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='Confidence Interval'
        ))
        
        fig.update_layout(
            title=f"Prophet Forecast for {ticker} ({periods} days)",
            xaxis_title="Date",
            yaxis_title="Price",
            showlegend=True,
            height=600,
            hovermode='x unified',
            plot_bgcolor='white',
            xaxis=dict(
                showgrid=True, 
                gridwidth=1, 
                gridcolor='#f0f0f0',
                type='date',
                dtick="M1",
                tickformat="%b %Y"
            ),
            yaxis=dict(
                showgrid=True, 
                gridwidth=1, 
                gridcolor='#f0f0f0',
                tickprefix="$"
            )
        )
        
        return forecast, fig
        
    except Exception as e:
        st.error(f"Error in Prophet prediction: {e}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        raise e

def predict_with_arima(data, periods):
    # Prepare data
    try:
        df_arima = data['Close'].values
        
        # Fit ARIMA model
        model = ARIMA(df_arima, order=(5, 1, 0))  # Parameters can be tuned
        model_fit = model.fit()
        
        # Forecast with confidence intervals
        forecast = model_fit.get_forecast(steps=periods)
        mean_forecast = forecast.predicted_mean
        confidence_intervals = forecast.conf_int()
        forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=periods)
        
        # Create a dataframe for forecast
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Predicted': mean_forecast,
            'Lower': confidence_intervals[:, 0],  # Using direct numpy indexing instead of iloc
            'Upper': confidence_intervals[:, 1]   # Using direct numpy indexing instead of iloc
        })
        
        # Plot
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Historical Data',
            line=dict(color='#1f77b4')
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Predicted'],
            mode='lines',
            name='ARIMA Forecast',
            line=dict(color='#ff7f0e')
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df['Date'], forecast_df['Date'].iloc[::-1]]),
            y=pd.concat([forecast_df['Upper'], forecast_df['Lower'].iloc[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='95% Confidence Interval'
        ))
        
        fig.update_layout(
            title={
                'text': f"ARIMA Forecast for {ticker} ({periods} days)",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Date",
            yaxis_title="Price ($)",
            showlegend=True,
            height=600,
            hovermode='x unified',
            plot_bgcolor='white',
            xaxis=dict(
                showgrid=True, 
                gridwidth=1, 
                gridcolor='#f0f0f0',
                type='date',
                dtick="M1",
                tickformat="%b %Y"
            ),
            yaxis=dict(
                showgrid=True, 
                gridwidth=1, 
                gridcolor='#f0f0f0',
                tickprefix="$"
            )
        )
        
        return forecast_df, fig
    except Exception as e:
        st.error(f"Error in ARIMA prediction: {e}")
        raise e

def predict_with_ml(data, periods, model_type='lr'):
    try:
        # Prepare data
        df = data[['Close']].copy()
        
        # Create features (lagged values)
        for i in range(1, 6):
            df[f'lag_{i}'] = df['Close'].shift(i)
        
        df.dropna(inplace=True)
        
        # Split data
        X = df.drop('Close', axis=1)
        y = df['Close']
        
        # Scale features
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
        
        # Train model
        if model_type == 'lr':
            model = LinearRegression()
            model_name = "Linear Regression"
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model_name = "Random Forest"
        
        model.fit(X_scaled, y_scaled.ravel())
        
        # Make prediction
        predictions = []
        last_values = X.iloc[-1].values.reshape(1, -1)
        
        for _ in range(periods):
            next_pred_scaled = model.predict(scaler_X.transform(last_values))
            next_pred = scaler_y.inverse_transform(next_pred_scaled.reshape(-1, 1))[0][0]
            predictions.append(next_pred)
            
            # Update features for next prediction
            last_values = np.roll(last_values, 1)
            last_values[0][0] = next_pred
        
        # Create prediction dataframe
        forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=periods)
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Predicted': predictions
        })
        
        # Add prediction interval (simple estimate)
        if model_type == 'rf':
            # For Random Forest, create a simple confidence interval
            forecast_df['Lower'] = forecast_df['Predicted'] * 0.95
            forecast_df['Upper'] = forecast_df['Predicted'] * 1.05
        else:
            # For Linear Regression, create a simple confidence interval
            forecast_df['Lower'] = forecast_df['Predicted'] * 0.90
            forecast_df['Upper'] = forecast_df['Predicted'] * 1.10
        
        # Plot
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Predicted'],
            mode='lines',
            name=f'{model_name} Forecast',
            line=dict(color='red')
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df['Date'], forecast_df['Date'].iloc[::-1]]),
            y=pd.concat([forecast_df['Upper'], forecast_df['Lower'].iloc[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='Confidence Interval'
        ))
        
        fig.update_layout(
            title=f"{model_name} Forecast for {ticker} ({periods} days)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            showlegend=True,
            height=600,
            hovermode='x unified',
            template='plotly_white',
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                type='date',
                dtick="M1",
                tickformat="%b %Y"
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                tickprefix="$"
            )
        )
        
        return forecast_df, fig
    except Exception as e:
        st.error(f"Error in {model_type} prediction: {e}")
        raise e

# Add Black-Scholes option pricing functions
def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate Black-Scholes call option price
    
    Parameters:
    S: current stock price
    K: strike price
    T: time to maturity in years
    r: risk-free interest rate
    sigma: volatility
    
    Returns:
    Call option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    return call

def black_scholes_put(S, K, T, r, sigma):
    """
    Calculate Black-Scholes put option price
    
    Parameters:
    S: current stock price
    K: strike price
    T: time to maturity in years
    r: risk-free interest rate
    sigma: volatility
    
    Returns:
    Put option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
    return put

def create_option_price_heatmap(option_type, min_spot, max_spot, min_vol, max_vol, K, T, r):
    """
    Create a heatmap of option prices for different spot prices and volatilities
    
    Parameters:
    option_type: 'call' or 'put'
    min_spot: minimum spot price
    max_spot: maximum spot price
    min_vol: minimum volatility
    max_vol: maximum volatility
    K: strike price
    T: time to maturity
    r: risk-free rate
    
    Returns:
    Plotly figure
    """
    # Create meshgrid of spot prices and volatilities
    # Use fewer points to make text more readable
    num_points = 15  # Reduced from 30 to make text readable
    spot_prices = np.linspace(min_spot, max_spot, num_points)
    volatilities = np.linspace(min_vol, max_vol, num_points)
    S_mesh, sigma_mesh = np.meshgrid(spot_prices, volatilities)
    
    # Calculate option prices
    if option_type == 'call':
        option_prices = np.vectorize(black_scholes_call)(S_mesh, K, T, r, sigma_mesh)
        # For call options: profit when S > K (in-the-money)
        profit_threshold = K
        title = 'Call Option Prices (Green = Potential Profit, Red = Potential Loss)'
    else:  # put
        option_prices = np.vectorize(black_scholes_put)(S_mesh, K, T, r, sigma_mesh)
        # For put options: profit when K > S (in-the-money)
        profit_threshold = K
        title = 'Put Option Prices (Green = Potential Profit, Red = Potential Loss)'

    # Create custom colorscale based on the option type
    if option_type == 'call':
        # For call options: green on right side (spot > strike), red on left (spot < strike)
        colorscale = [
            [0, 'rgba(220, 0, 0, 1)'],         # Deep red for lowest values
            [K/max_spot * 0.9, 'rgba(255, 100, 100, 1)'],  # Light red near strike
            [K/max_spot, 'rgba(255, 255, 255, 1)'],     # White at strike price
            [K/max_spot * 1.1, 'rgba(100, 200, 100, 1)'],  # Light green just above strike
            [1, 'rgba(0, 150, 0, 1)']          # Deep green for highest values
        ]
    else:
        # For put options: green on left side (spot < strike), red on right (spot > strike)
        colorscale = [
            [0, 'rgba(0, 150, 0, 1)'],         # Deep green for lowest values
            [K/max_spot * 0.9, 'rgba(100, 200, 100, 1)'],  # Light green near strike
            [K/max_spot, 'rgba(255, 255, 255, 1)'],     # White at strike price
            [K/max_spot * 1.1, 'rgba(255, 100, 100, 1)'],  # Light red just above strike
            [1, 'rgba(220, 0, 0, 1)']          # Deep red for highest values
        ]
    
    # Create a DataFrame for annotations and hover text
    option_prices_flat = option_prices.flatten()
    df = pd.DataFrame({
        'Spot Price': S_mesh.flatten(),
        'Volatility': sigma_mesh.flatten(),
        'Option Price': option_prices_flat,
        'Price Text': [f"${price:.2f}" for price in option_prices_flat]
    })
    
    # Create the heatmap figure
    fig = go.Figure()
    
    # Add main heatmap with custom colors
    fig.add_trace(
        go.Heatmap(
            x=spot_prices,
            y=volatilities,
            z=option_prices,
            colorscale=colorscale,
            colorbar=dict(
                title='Option Price ($)',
                tickprefix='$'
            ),
            text=[[f"${val:.2f}" for val in row] for row in option_prices],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='Spot Price: $%{x:.2f}<br>Volatility: %{y:.1%}<br>Option Price: %{text}<extra></extra>'
        )
    )
    
    # Add a vertical line at the strike price
    fig.add_shape(
        type="line",
        x0=K, y0=min_vol,
        x1=K, y1=max_vol,
        line=dict(color="black", width=2, dash="dash"),
    )
    
    # Add annotation for strike price
    fig.add_annotation(
        x=K,
        y=max_vol * 0.95,
        text=f"Strike Price: ${K:.2f}",
        showarrow=True,
        arrowhead=1,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="black",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    # Add region labels
    if option_type == 'call':
        # For call options
        fig.add_annotation(
            x=(min_spot + K)/2,
            y=max_vol * 0.8,
            text="Out-of-Money<br>Potential Loss",
            showarrow=False,
            font=dict(color="white", size=12),
            bgcolor="rgba(220,0,0,0.7)",
            bordercolor="rgba(220,0,0,0.7)",
            borderwidth=1,
            borderpad=4
        )
        fig.add_annotation(
            x=(max_spot + K)/2,
            y=max_vol * 0.8,
            text="In-the-Money<br>Potential Profit",
            showarrow=False,
            font=dict(color="white", size=12),
            bgcolor="rgba(0,150,0,0.7)",
            bordercolor="rgba(0,150,0,0.7)",
            borderwidth=1,
            borderpad=4
        )
    else:
        # For put options
        fig.add_annotation(
            x=(min_spot + K)/2,
            y=max_vol * 0.8,
            text="In-the-Money<br>Potential Profit",
            showarrow=False,
            font=dict(color="white", size=12),
            bgcolor="rgba(0,150,0,0.7)",
            bordercolor="rgba(0,150,0,0.7)",
            borderwidth=1,
            borderpad=4
        )
        fig.add_annotation(
            x=(max_spot + K)/2,
            y=max_vol * 0.8,
            text="Out-of-Money<br>Potential Loss",
            showarrow=False,
            font=dict(color="white", size=12),
            bgcolor="rgba(220,0,0,0.7)",
            bordercolor="rgba(220,0,0,0.7)",
            borderwidth=1,
            borderpad=4
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=600,
        template='plotly_white',
        yaxis=dict(
            tickformat='.0%',
            title='Volatility (σ)'
        ),
        xaxis=dict(
            tickprefix='$',
            title='Spot Price ($)'
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

# Main application
try:
    # Load data
    data = load_data(ticker, start_date, end_date)
    
    if data is not None:
        # Display data info
        st.write(f"### {ticker} Stock Data")
        st.write(f"Data from {start_date} to {end_date}")
        
        # Get detailed stock information
        stock_info = get_detailed_stock_info(ticker)
        
        if stock_info:
            st.subheader("Stock Information")
            
            # Create multi-column layout for detailed metrics
            col1, col2, col3 = st.columns(3)
            
            # Create a list of all metrics for display
            all_metrics = [
                ("Market Cap", stock_info.get("Market Cap", "N/A")),
                ("Revenue (ttm)", stock_info.get("Revenue (ttm)", "N/A")),
                ("Net Income (ttm)", stock_info.get("Net Income (ttm)", "N/A")),
                ("Shares Out", stock_info.get("Shares Out", "N/A")),
                ("EPS (ttm)", stock_info.get("EPS (ttm)", "N/A")),
                ("PE Ratio", stock_info.get("PE Ratio", "N/A")),
                ("Forward PE", stock_info.get("Forward PE", "N/A")),
                ("Dividend", stock_info.get("Dividend", "N/A")),
                ("Ex-Dividend Date", stock_info.get("Ex-Dividend Date", "N/A")),
                ("Volume", stock_info.get("Volume", "N/A")),
                ("Open", stock_info.get("Open", "N/A")),
                ("Previous Close", stock_info.get("Previous Close", "N/A")),
                ("Day's Range", stock_info.get("Day's Range", "N/A")),
                ("52-Week Range", stock_info.get("52-Week Range", "N/A")),
                ("Beta", stock_info.get("Beta", "N/A")),
                ("Analysts", stock_info.get("Analysts", "N/A")),
                ("Price Target", stock_info.get("Price Target", "N/A"))
            ]
            
            # Display metrics in three columns
            metrics_per_column = len(all_metrics) // 3
            
            for i, (label, value) in enumerate(all_metrics):
                if i < metrics_per_column:
                    with col1:
                        st.metric(label, value)
                elif i < metrics_per_column * 2:
                    with col2:
                        st.metric(label, value)
                else:
                    with col3:
                        st.metric(label, value)
        
        # Display tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📊 Technical Analysis", "🔮 Predictions", "💰 Options Pricing"])
        
        with tab1:
            st.subheader("Stock Overview")
            
            # Add a small section for date range info with error handling
            try:
                start_str = start_date.strftime("%b %d, %Y")
                end_str = end_date.strftime("%b %d, %Y")
                days_diff = (data.index[-1] - data.index[0]).days
                st.info(f"📅 Data available from {start_str} to {end_str} ({days_diff} days)")
            except:
                st.info(f"📅 Data available from {start_date} to {end_date}")
            
            # Display the data table directly
            st.subheader("Recent Trading Data")
            st.dataframe(data.tail(10), use_container_width=True)
            st.caption("Showing the 10 most recent data points")
            
            # Add download button for the complete dataset
            csv = data.to_csv()
            st.download_button(
                label="Download Complete Dataset",
                data=csv,
                file_name=f'{ticker}_historical_data.csv',
                mime='text/csv',
            )
        
        with tab2:
            st.subheader("Technical Indicators")
            plot_technical_indicators(data)
        
        with tab3:
            st.subheader("Stock Price Prediction")
            st.write(f"Prediction for the next {prediction_days} days using Prophet model")
            
            # Run prediction using Prophet only
            with st.spinner(f'Running Prophet model prediction...'):
                try:
                    st.info("Running Prophet model...")
                    forecast, fig = predict_with_prophet(data, prediction_days)
                    st.plotly_chart(fig, use_container_width=True)
                    st.write("Prediction Results:")
                    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(prediction_days).reset_index(drop=True))
                    
                    st.success(f'Prediction completed!')
                    
                    # Download prediction as CSV
                    download_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(prediction_days).reset_index(drop=True)
                    download_df.columns = ['Date', 'Predicted', 'Lower Bound', 'Upper Bound']
                    csv = download_df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download Prediction Data",
                        data=csv,
                        file_name=f'{ticker}_prediction_Prophet.csv',
                        mime='text/csv',
                    )
                except Exception as model_error:
                    st.error(f"Error in prediction: {str(model_error)}")
                    import traceback
                    st.error(f"Detailed error: {traceback.format_exc()}")
                    st.info("Try with a smaller prediction period.")
            
        with tab4:
            st.subheader("Options Pricing Analysis")
            
            # Display option pricing parameters
            st.write("### Option Pricing Parameters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Asset Price", f"${current_price:.2f}")
                st.metric("Strike Price", f"${strike_price:.2f}")
            
            with col2:
                st.metric("Time to Maturity", f"{time_to_maturity:.2f} years")
                st.metric("Risk-Free Rate", f"{risk_free_rate:.2%}")
            
            with col3:
                st.metric("Volatility", f"{volatility:.2%}")
                call_price = black_scholes_call(current_price, strike_price, time_to_maturity, risk_free_rate, volatility)
                put_price = black_scholes_put(current_price, strike_price, time_to_maturity, risk_free_rate, volatility)
                st.metric("Call Option Price", f"${call_price:.2f}")
                st.metric("Put Option Price", f"${put_price:.2f}")
            
            # Option price vs. spot price
            st.write("### Option Price vs. Spot Price")
            spot_range = np.linspace(min_spot, max_spot, 100)
            call_prices = [black_scholes_call(s, strike_price, time_to_maturity, risk_free_rate, volatility) for s in spot_range]
            put_prices = [black_scholes_put(s, strike_price, time_to_maturity, risk_free_rate, volatility) for s in spot_range]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spot_range, y=call_prices, mode='lines', name='Call Option', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=spot_range, y=put_prices, mode='lines', name='Put Option', line=dict(color='red')))
            
            # Add vertical line at current price
            fig.add_vline(x=current_price, line_dash="dash", line_color="blue", annotation_text=f"Current: ${current_price}")
            
            # Add horizontal line at strike price
            fig.add_hline(y=strike_price - current_price, line_dash="dash", line_color="gray", annotation_text=f"Strike: ${strike_price}")
            
            fig.update_layout(
                title="Option Prices vs. Spot Price",
                xaxis_title="Spot Price ($)",
                yaxis_title="Option Price ($)",
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create heatmaps
            st.write("### Option Price Heatmaps")
            heatmap_tab1, heatmap_tab2 = st.tabs(["Call Option Heatmap", "Put Option Heatmap"])
            
            with heatmap_tab1:
                call_heatmap = create_option_price_heatmap(
                    'call', min_spot, max_spot, min_vol, max_vol, 
                    strike_price, time_to_maturity, risk_free_rate
                )
                st.plotly_chart(call_heatmap, use_container_width=True)
                st.write("""
                This heatmap shows call option prices for different combinations of spot prices and volatilities.
                Higher spot prices and higher volatilities generally lead to higher call option prices.
                """)
            
            with heatmap_tab2:
                put_heatmap = create_option_price_heatmap(
                    'put', min_spot, max_spot, min_vol, max_vol, 
                    strike_price, time_to_maturity, risk_free_rate
                )
                st.plotly_chart(put_heatmap, use_container_width=True)
                st.write("""
                This heatmap shows put option prices for different combinations of spot prices and volatilities.
                Lower spot prices and higher volatilities generally lead to higher put option prices.
                """)
    else:
        st.warning("Please enter a valid ticker symbol.")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Try with a different ticker or date range.")

st.sidebar.markdown("---")
st.sidebar.info("""
This app uses historical stock data to predict future prices using the Prophet model.
Note: These predictions are for educational purposes only and should not be used for financial decisions.
""")

# Run the app with: streamlit run app.py

