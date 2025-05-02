
import streamlit as st
import pandas as pd
from alpaca_trade_api.rest import REST
import ta
import altair as alt
import yfinance as yf


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
from datetime import datetime, timedelta
import pytz

def get_intraday_data(symbol='SPY', interval='1Min', limit=100):
    now = datetime.now(pytz.timezone("America/New_York"))
    start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end = now

    bars = api.get_bars(
        symbol,
        timeframe=interval,
        start=start.isoformat(),
        end=end.isoformat()
    )

    if hasattr(bars, 'df'):
        df = bars.df
        df.index = pd.to_datetime(df.index).tz_convert('America/New_York')
        return df
    else:
        return pd.DataFrame()

# Compute signal scores
def compute_signals(df, mode="full"):
    import ta

    df = df.copy()

    if mode == "short":
        df["weight"] = [0.1 * (i + 1) for i in range(len(df))]
        df["weight"] /= df["weight"].sum()  # normalize to sum to 1

    # Use shorter indicators for zoomed-in view
    if mode == "short":
        rsi_period = 7
        ma_fast = 5
        ma_slow = 10
    else:
        rsi_period = 14
        ma_fast = 5
        ma_slow = 20

    # Calculate indicators
    try:
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=rsi_period).rsi()
        df['ma_fast'] = df['close'].rolling(ma_fast).mean()
        df['ma_slow'] = df['close'].rolling(ma_slow).mean()
        df = df.dropna()
    except Exception:
        return {
            'order_flow': 0.5,
            'quant_model': 0.5,
            'market_microstructure': 0.5,
            'momentum_technical': 0.5,
            'volume_liquidity_heatmaps': 0.5,
            'gamma_options_flow': 0.5,
            'machine_learning_prediction': 0.5,
            'news_sentiment': 0.5,
            'random_noise': 0.5
        }
    
    # Weighted momentum example (for "short" mode)
    if mode == "short" and 'weight' in df.columns:
        try:
            # Compare recent MA crossovers, weighted
            ma_diff = df['ma_fast'] - df['ma_slow']
            momentum_score = (ma_diff * df['weight']).sum()
            momentum_score = 1.0 if momentum_score > 0 else 0.0
        except:
            momentum_score = 0.5
    else:
        momentum_score = 1.0 if df['ma_fast'].iloc[-1] > df['ma_slow'].iloc[-1] else 0.3

    if df.empty:
        return {}, pd.DataFrame()

    latest = df.iloc[-1]

    return {
        'order_flow': 0.7,
        'quant_model': 0.6,
        'market_microstructure': 0.6,
        'momentum_technical': momentum_score,
        'volume_liquidity_heatmaps': 0.4,
        'gamma_options_flow': 0.5,
        'machine_learning_prediction': 0.3,
        'news_sentiment': 0.2,
        'random_noise': 0.1
    }, df

# Score interpretation
def calculate_direction_score(weights, scores):
    return sum(weights[k] * scores.get(k, 0) for k in weights)

def interpret_score(score):
    if score > 0.6:
        return "↑ Bullish Bias"
    elif score < 0.4:
        return "↓ Bearish Bias"
    return "→ Sideways / Neutral"

def get_options_flow(symbol):
    import yfinance as yf
    import pandas as pd

    try:
        ticker = yf.Ticker(symbol)
        exp_dates = ticker.options

        if not exp_dates:
            raise ValueError(f"No expiration dates found for symbol: {symbol}")

        chain = ticker.option_chain(exp_dates[0])
        calls = chain.calls
        puts = chain.puts

        calls["Type"] = "call"
        puts["Type"] = "put"

        all_options = pd.concat([calls, puts], ignore_index=True)
        return all_options

    except Exception as e:
        raise ValueError(f"Failed to retrieve option chain for {symbol}: {e}")

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

from statistics import mean

def get_options_flow(symbol):
    try:
        ticker = yf.Ticker(symbol)
        exp_dates = ticker.options

        if not exp_dates:
            return {"error": "No expirations found"}

        # Use nearest expiration
        chain = ticker.option_chain(exp_dates[0])
        calls = chain.calls
        puts = chain.puts

        call_volume = calls['volume'].sum()
        put_volume = puts['volume'].sum()

        bullish_ratio = call_volume / (call_volume + put_volume + 1e-6)  # avoid div by 0

        avg_call_iv = mean(calls['impliedVolatility'].dropna())
        avg_put_iv = mean(puts['impliedVolatility'].dropna())

        # Create a basic scoring model (you can improve this)
        score = 0.5 + (bullish_ratio - 0.5) * 1.5  # boost if skewed bullish

        return {
            "call_volume": int(call_volume),
            "put_volume": int(put_volume),
            "bullish_ratio": round(bullish_ratio, 2),
            "avg_call_iv": round(avg_call_iv, 3),
            "avg_put_iv": round(avg_put_iv, 3),
            "suggested_signal_score": round(score, 2)
        }

    except Exception as e:
        return {"error": str(e)}

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
            data_for_signals = df.tail(20) if view_option == "Last 20 Bars (Zoomed In)" else df
            mode = "short" if view_option == "Last 20 Bars (Zoomed In)" else "full"
            scores, enriched_df = compute_signals(data_for_signals, mode=mode)

            # ✅ These must come AFTER data_for_signals is defined
            st.markdown("### Signal Insights")
            if enriched_df is not None and 'rsi' in enriched_df.columns and not enriched_df['rsi'].isna().all():
                st.markdown(f"** RSI: `{enriched_df['rsi'].iloc[-1]:.2f}`")
            else:
                st.markdown("** RSI: N/A")

            if 'ma_fast' in enriched_df.columns and 'ma_slow' in enriched_df.columns:
                st.markdown(f"** MAS vs MA20: `{enriched_df['ma_fast'].iloc[-1]:.2f}` vs `{enriched_df['ma_slow'].iloc[-1]:.2f}`")

            signal_weights = {
                "short": {
                    "momentum_technical": 0.5,
                    "volume_liquidity_heatmaps": 0.3,
                    "random_noise": 0.2
                },
                "full": {
                    "order_flow": 0.3,
                    "quant_model": 0.3,
                    "market_microstructure": 0.2,
                    "momentum_technical": 0.1,
                    "volume_liquidity_heatmaps": 0.1
                }
            }

            score = calculate_direction_score(signal_weights[mode], scores)
            bias = interpret_score(score)

            st.metric("Prediction Score", f"{score:.2f}")
            st.subheader(f"Market Bias: {bias}")

            # Fetch and display options flow from Yahoo Finance
            with st.expander("📊 Options Flow Snapshot (Yahoo Finance)", expanded=False):
                try:
                    options_data = get_options_flow(symbol)
                    if isinstance(options_data, pd.DataFrame) and 'Type' in options_data.columns and not options_data.empty:
                        st.write("Calls (Volume > 0):")
                        st.dataframe(options_data[options_data['Type'] == 'call'][['Strike', 'Last Price', 'Bid', 'Ask', 'Volume']])

                        st.write("Puts (Volume > 0):")
                        st.dataframe(options_data[options_data['Type'] == 'put'][['Strike', 'Last Price', 'Bid', 'Ask', 'Volume']])
                    else:
                        st.error(f"Options flow data not available or malformed for: {symbol}")
                except Exception as e:
                    st.error(f"Options flow data fetch error for {symbol}: {e}")

            # Chart logic based on dropdown selection
            if view_option == "Last 20 Bars (Zoomed In)":
                plot_close_chart(df, zoom=True)
            else:
                plot_close_chart(df)

            # Optional insights for user
            st.markdown("### Signal Insights")
            if 'rsi' in data_for_signals.columns:
                st.markdown(f"** RSI: `{data_for_signals['rsi'].iloc[-1]:.2f}`")

            if 'ma_fast' in data_for_signals.columns and 'ma_slow' in data_for_signals.columns:
                st.markdown(f"** MAS vs MA20: `{data_for_signals['ma_fast'].iloc[-1]:.2f}` vs `{data_for_signals['ma_slow'].iloc[-1]:.2f}`")

            # Enhanced Alerts Based on Score Ranges
            if score > 0.8:
                st.success("🔥 VERY STRONG BULLISH SIGNAL!")
                st.audio("https://www.soundjay.com/button/beep-09.wav", autoplay=True)
            elif score >= 0.68:
                st.success("🟢 Moderate Bullish Signal")
                st.audio("https://www.soundjay.com/button/beep-07.wav", autoplay=True)
            elif score >= 0.4:
                st.info("⚪️ Neutral Market - No strong trend")
            elif score >= 0.25:
                st.warning("🔴 Moderate Bearish Signal")
                st.audio("https://www.soundjay.com/button/beep-08b.wav", autoplay=True)
            else:
                st.error("⚡ VERY STRONG BEARISH SIGNAL!")
                st.audio("https://www.soundjay.com/button/beep-05.wav", autoplay=True)



