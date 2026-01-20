import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ------------------- CONFIG -------------------
st.set_page_config(
    page_title="Apple Stock Forecast Dashboard",
    page_icon="🍎",
    layout="wide",
)

# ------------------- STYLING -------------------
st.markdown(
    """
    <style>
    .big-title {
        font-size: 38px !important;
        font-weight: 800;
        color: #ffffff;
        padding: 10px 0px;
    }
    .sub-title {
        font-size: 16px !important;
        color: #d0d0d0;
        margin-top: -10px;
    }
    .kpi-card {
        background-color: #111827;
        padding: 16px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
    }
    .kpi-label {
        color: #9CA3AF;
        font-size: 13px;
        font-weight: 500;
    }
    .kpi-value {
        color: #ffffff;
        font-size: 24px;
        font-weight: 800;
        margin-top: 5px;
    }
    .kpi-small {
        color: #9CA3AF;
        font-size: 12px;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------- DATA PATH -------------------
DATA_PATH = "AAPL (5).csv"   # dataset must be in same folder

# ------------------- FUNCTIONS -------------------
def clean_columns(df):
    df.columns = df.columns.astype(str)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True)
    df = df.loc[:, ~df.columns.duplicated()]  # remove duplicates
    return df

def detect_date_column(df):
    possible = ["date", "datetime", "timestamp", "time"]
    for col in df.columns:
        c = col.lower().replace(" ", "").replace("_", "")
        if c in possible:
            return col
    return None

def detect_target_column(df):
    for col in df.columns:
        c = col.lower().replace(" ", "").replace("_", "")
        if c in ["adjclose", "adj_close", "adjustedclose", "adjustedcloseprice"]:
            return col
    for col in df.columns:
        if col.lower().strip() == "close":
            return col
    return None

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mp = mape(y_true, y_pred)
    return rmse, mae, mp

def add_moving_averages(df, target_col):
    df["MA20"] = df[target_col].rolling(20).mean()
    df["MA50"] = df[target_col].rolling(50).mean()
    df["MA200"] = df[target_col].rolling(200).mean()
    return df

def plot_line(series, title, ylabel):
    fig = plt.figure(figsize=(12, 4))
    plt.plot(series.index, series.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.grid(True)
    st.pyplot(fig)

# ------------------- HEADER -------------------
st.markdown('<div class="big-title">🍎 Apple Stock Forecast Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">EDA • Trend Insights • Returns & Volatility • ARIMA Forecast (Next 30 Business Days)</div>', unsafe_allow_html=True)

st.divider()

# ------------------- SIDEBAR SETTINGS -------------------
st.sidebar.header("⚙️ Forecast Settings")

order_p = st.sidebar.slider("ARIMA p", 0, 10, 5)
order_d = st.sidebar.slider("ARIMA d", 0, 2, 1)
order_q = st.sidebar.slider("ARIMA q", 0, 10, 0)

forecast_days = st.sidebar.slider("Forecast Days", 7, 90, 30)

show_data = st.sidebar.checkbox("Show raw dataset", value=False)

# ------------------- LOAD DATA -------------------
try:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
except Exception as e:
    st.error(f"❌ Dataset not found: '{DATA_PATH}'\n\n➡️ Put your CSV in same folder as app.py")
    st.stop()

df = clean_columns(df)

date_col = detect_date_column(df)
target_col = detect_target_column(df)

if date_col is None:
    st.error("❌ Date column not found in dataset. Make sure file contains Date.")
    st.stop()

if target_col is None:
    st.error("❌ Target column not found. Dataset should contain Adj_Close or Close.")
    st.stop()

df.rename(columns={date_col: "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df.dropna(subset=[target_col])

# fix business frequency
df = df.asfreq("B")
df[target_col] = df[target_col].ffill()

# create features
df["Daily_Return"] = df[target_col].pct_change()
df["Volatility_20"] = df["Daily_Return"].rolling(20).std() * np.sqrt(252)
df = add_moving_averages(df, target_col)

ts = df[target_col].dropna()

# ------------------- KPI SECTION -------------------
latest_price = ts.iloc[-1]
start_price = ts.iloc[0]
overall_return = ((latest_price - start_price) / start_price) * 100

max_price = ts.max()
min_price = ts.min()
avg_price = ts.mean()

col1, col2, col3, col4 = st.columns(4)

col1.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">Latest Price</div>
    <div class="kpi-value">${latest_price:,.2f}</div>
    <div class="kpi-small">Last trading day</div>
</div>
""", unsafe_allow_html=True)

col2.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">Overall Return</div>
    <div class="kpi-value">{overall_return:.2f}%</div>
    <div class="kpi-small">From first record → latest</div>
</div>
""", unsafe_allow_html=True)

col3.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">All-time High</div>
    <div class="kpi-value">${max_price:,.2f}</div>
    <div class="kpi-small">Peak value in dataset</div>
</div>
""", unsafe_allow_html=True)

col4.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">All-time Low</div>
    <div class="kpi-value">${min_price:,.2f}</div>
    <div class="kpi-small">Lowest value in dataset</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ------------------- DATA PREVIEW -------------------
if show_data:
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.tail(20), use_container_width=True)

# ------------------- EDA VISUALS -------------------
st.subheader("📊 Visual Insights (EDA)")

tab1, tab2, tab3, tab4 = st.tabs(["Price Trend", "Moving Averages", "Returns", "Volatility"])

with tab1:
    st.write("### Price Trend")
    fig = plt.figure(figsize=(12, 5))
    plt.plot(ts.index, ts.values)
    plt.title("Stock Price Over Time")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.grid(True)
    st.pyplot(fig)

with tab2:
    st.write("### Moving Average Strategy")
    fig = plt.figure(figsize=(12, 5))
    plt.plot(df.index, df[target_col], label="Price")
    plt.plot(df.index, df["MA20"], label="MA20")
    plt.plot(df.index, df["MA50"], label="MA50")
    plt.plot(df.index, df["MA200"], label="MA200")
    plt.title("Price with Moving Averages (20/50/200)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)

with tab3:
    st.write("### Daily Returns Analysis")

    # returns line
    fig = plt.figure(figsize=(12, 4))
    plt.plot(df.index, df["Daily_Return"], alpha=0.9)
    plt.title("Daily Returns Over Time")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.grid(True)
    st.pyplot(fig)

    # returns distribution
    fig = plt.figure(figsize=(12, 4))
    plt.hist(df["Daily_Return"].dropna(), bins=50)
    plt.title("Distribution of Daily Returns")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.grid(True)
    st.pyplot(fig)

with tab4:
    st.write("### Volatility (20-Day Rolling Annualized)")
    fig = plt.figure(figsize=(12, 4))
    plt.plot(df.index, df["Volatility_20"])
    plt.title("20-Day Rolling Volatility (Annualized)")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.grid(True)
    st.pyplot(fig)

# ------------------- VOLUME INSIGHTS -------------------
if "Volume" in df.columns:
    st.subheader("📌 Volume vs Price Insight")
    c1, c2 = st.columns(2)

    with c1:
        fig = plt.figure(figsize=(10, 4))
        plt.plot(df.index, df["Volume"])
        plt.title("Trading Volume Over Time")
        plt.xlabel("Date")
        plt.ylabel("Volume")
        plt.grid(True)
        st.pyplot(fig)

    with c2:
        fig = plt.figure(figsize=(10, 4))
        plt.scatter(df["Volume"], df[target_col], alpha=0.4)
        plt.title("Volume vs Price Relationship")
        plt.xlabel("Volume")
        plt.ylabel("Price")
        plt.grid(True)
        st.pyplot(fig)

st.divider()

# ------------------- FORECASTING -------------------
st.subheader("🔮 ARIMA Forecasting")

split_idx = int(len(ts) * 0.8)
train, test = ts.iloc[:split_idx], ts.iloc[split_idx:]

st.write(f"✅ Train: **{len(train)}** records | Test: **{len(test)}** records")

try:
    model = ARIMA(train, order=(order_p, order_d, order_q))
    fit = model.fit()
except Exception as e:
    st.error(f"❌ ARIMA Model error: {e}")
    st.stop()

pred_test = fit.forecast(steps=len(test))
rmse, mae, mp = eval_metrics(test, pred_test)

m1, m2, m3 = st.columns(3)
m1.metric("RMSE", f"{rmse:.4f}")
m2.metric("MAE", f"{mae:.4f}")
m3.metric("MAPE (%)", f"{mp:.2f}")

# plot test forecast
fig = plt.figure(figsize=(12, 5))
plt.plot(test.index, test.values, label="Actual")
plt.plot(test.index, pred_test.values, label="Forecast")
plt.title("Test Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
st.pyplot(fig)

# future forecast
future_forecast = fit.forecast(steps=forecast_days)
future_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq="B")

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast_Price": future_forecast.values
})

# plot future forecast
fig = plt.figure(figsize=(12, 5))
plt.plot(ts.index[-250:], ts.values[-250:], label="Historical")
plt.plot(forecast_df["Date"], forecast_df["Forecast_Price"], label="Future Forecast")
plt.title(f"Next {forecast_days} Business Days Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
st.pyplot(fig)

st.subheader("📌 Forecast Output")
st.dataframe(forecast_df, use_container_width=True)

csv = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Forecast CSV",
    data=csv,
    file_name="apple_forecast.csv",
    mime="text/csv"
)

st.success("✅ Dashboard loaded successfully!")
