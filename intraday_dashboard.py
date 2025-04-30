
import streamlit as st
import pandas as pd
from alpaca_trade_api.rest import REST
import ta
import altair as alt

# Streamlit Secrets for Alpaca API keys
API_KEY = st.secrets["API_KEY"]
API_SECRET = st.secrets["API_SECRET"]
MODE = st.secrets.get("MODE", "paper")

if MODE == "paper":
    BASE_URL = "https://paper-api.alpaca.markets"
else:
    BASE_URL = "https://api.alpaca.markets"


api = REST(API_KEY, API_SECRET, base_url=BASE_URL, api_version='v2')

# Signal weightings
signal_weights = {
    "order_flow": 0.30,
    "quant_model": 0.20,
    "market_microstructure": 0.15,
    "momentum_technical": 0.10,
    "volume_liquidity_heatmaps": 0.08,
    "gamma_options_flow": 0.07,
    "machine_learning_prediction": 0.05,
    "news_sentiment": 0.03,
    "random_noise": 0.02
}

# Get recent data
def get_intraday_data(symbol='SPY', interval='1Min', limit=100):
    bars = api.get_bars(symbol, timeframe=interval, limit=limit)

    # Safely convert to DataFrame
    if hasattr(bars, 'df'):
        df = bars.df
        df.index = pd.to_datetime(df.index)  # optional, for clarity
        return df
    else:
        return pd.DataFrame()  # fallback in rare cases

# Compute signal scores
def compute_signals(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['pct_change'] = df['close'].pct_change()
    
    latest = df.iloc[-1]

    return {
        'order_flow': df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1],  # relative volume
        'quant_model': (latest['close'] - df['open'].iloc[-1]) / df['open'].iloc[-1],   # intraday return
        'market_microstructure': df['pct_change'].std() * 100,  # volatility signal
        'momentum_technical': 1.0 if latest['ma5'] > latest['ma20'] else 0.3,
        'volume_liquidity_heatmaps': df['volume'].tail(10).mean() / df['volume'].mean(),
        'gamma_options_flow': 0.5,  # placeholder, no real gamma input
        'machine_learning_prediction': latest['rsi'] / 100,  # simplistic approximation
        'news_sentiment': 0.2,  # placeholder
        'random_noise': 0.1
    }

# Score interpretation
def calculate_direction_score(weights, scores):
    return sum(weights[k] * scores.get(k, 0) for k in weights)

def interpret_score(score):
    if score > 0.6:
        return "↑ Bullish Bias"
    elif score < 0.4:
        return "↓ Bearish Bias"
    return "→ Sideways / Neutral"

def plot_close_chart(df, title="Price Movement", zoom=False):
    df = df.copy()
    df['timestamp'] = df.index.tz_convert('America/New_York')

    if zoom:
        df = df.tail(20)

    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('timestamp:T', title='Time'),
        y=alt.Y('close:Q', title='Price', scale=alt.Scale(zero=False)),
        tooltip=['timestamp:T', 'close:Q']
    ).properties(
        title=title,
        width=700,
        height=300
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

# UI
st.title("📈 Intraday Direction Prediction Dashboard")

symbol = st.text_input("Enter Ticker Symbol", value="SPY")

# Move chart view selector *above* the prediction button so it persists
view_option = st.selectbox(
    "Chart View",
    ("Last 20 Bars (Zoomed In)", "Full Session")
)

if st.button("Run Live Prediction"):
    with st.spinner("Fetching live data and calculating signals..."):
        df = get_intraday_data(symbol)

        if df.empty or 'close' not in df.columns:
            st.error("No valid intraday data returned. Please check the ticker symbol, market hours, or your API subscription level.")
        else:
            scores = compute_signals(df)
            score = calculate_direction_score(signal_weights, scores)
            bias = interpret_score(score)

            st.metric("Prediction Score", f"{score:.2f}")
            st.subheader(f"Market Bias: {bias}")

            # Chart logic based on dropdown selection
            if view_option == "Last 20 Bars (Zoomed In)":
                plot_close_chart(df, zoom=True)
            else:
                plot_close_chart(df)

            # Optional insights for user
            st.markdown("### Signal Insights")
            st.markdown(f"- RSI: `{df['rsi'].iloc[-1]:.2f}`")
            st.markdown(f"- MA5 vs MA20: `{df['ma5'].iloc[-1]:.2f}` vs `{df['ma20'].iloc[-1]:.2f}`")

            # Alerts
            if score > 0.7:
                st.warning("🚨 STRONG Bullish Signal!")
                st.audio("https://www.soundjay.com/button/beep-07.wav", autoplay=True)
            elif score < 0.3:
                st.warning("🚨 STRONG Bearish Signal!")
                st.audio("https://www.soundjay.com/button/beep-08b.wav", autoplay=True)

