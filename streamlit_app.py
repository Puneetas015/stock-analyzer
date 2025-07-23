import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

st.set_page_config(page_title="📈 Stock Analyzer", layout="wide")

st.title("📊 Stock Price Analyzer & Predictor")
st.markdown("Predict stock trend (Buy/Sell) using technical indicators + ML model.")

# Sidebar
st.sidebar.header("Select Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. RELIANCE.NS)", value="RELIANCE.NS")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

# Download data
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

data = load_data(ticker, start_date, end_date)

if data.empty:
    st.error("⚠️ No data found. Please check the ticker and date range.")
    st.stop()

st.subheader("📉 Raw Stock Data")
st.dataframe(data.tail(), use_container_width=True)

# Feature Engineering
def add_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df.dropna(inplace=True)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

data = add_technical_indicators(data)

# Show plots
st.subheader("📈 Price & SMA Indicators")
fig, ax = plt.subplots()
ax.plot(data['Close'], label='Close Price', color='black')
ax.plot(data['SMA_20'], label='SMA 20', color='blue', linestyle='--')
ax.plot(data['SMA_50'], label='SMA 50', color='green', linestyle='--')
ax.set_title(f"{ticker} Price with SMA Indicators")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# ML Section
st.subheader("🤖 Predict Buy/Sell Using Random Forest")

features = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD']
X = data[features]
y = data['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Predict
predictions = model.predict(X_scaled)

# Show prediction summary
data['Prediction'] = predictions
data['Signal'] = np.where(predictions == 1, 'Buy', 'Sell')

# Plot buy/sell signals
st.subheader("💹 Buy/Sell Signals")
fig2, ax2 = plt.subplots()
ax2.plot(data['Close'], label='Close Price', color='gray')
ax2.plot(data[data['Prediction'] == 1].index, data[data['Prediction'] == 1]['Close'], '^', markersize=10, color='green', label='Buy Signal')
ax2.plot(data[data['Prediction'] == 0].index, data[data['Prediction'] == 0]['Close'], 'v', markersize=10, color='red', label='Sell Signal')
ax2.set_title(f"{ticker} Buy/Sell Prediction")
ax2.legend()
st.pyplot(fig2)

# Show last prediction
last_signal = data['Signal'].iloc[-1]
st.success(f"📌 Last Predicted Signal: **{last_signal}** on {data.index[-1].date()}")

# Export Data
st.download_button("📥 Download Prediction Data", data.to_csv().encode('utf-8'), file_name=f"{ticker}_predictions.csv")

