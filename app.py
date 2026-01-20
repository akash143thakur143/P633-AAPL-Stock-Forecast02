import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Apple Stock Forecast Dashboard", layout="wide")

st.title("🍎 Apple Stock Forecast (Next 30 Days)")
st.write("This dashboard performs EDA + ARIMA forecasting using Adjusted Close Price.")

# ------------------ FUNCTIONS ------------------
def clean_columns(df):
    df.columns = df.columns.astype(str)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True)
    df = df.loc[:, ~df.columns.duplicated()]  # remove duplicate columns
    return df


def detect_date_column(df):
    possible_date_cols = ["date", "datetime", "timestamp", "time"]
    for col in df.columns:
        cleaned = col.lower().replace(" ", "").replace("_", "")
        if cleaned in possible_date_cols:
            return col
    # fallback: first column might be date
    return df.columns[0]


def detect_target_column(df):
    # Prefer Adj_Close
    for col in df.columns:
        c = col.lower().replace(" ", "").replace("_", "")
        if c in ["adjclose", "adj_close", "adjustedclose", "adjustedcloseprice"]:
            return col
    # fallback to Close
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


# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader("Upload Apple Stock CSV", type=["csv"])

order_p = st.sidebar.slider("ARIMA p", 0, 10, 5)
order_d = st.sidebar.slider("ARIMA d", 0, 2, 1)
order_q = st.sidebar.slider("ARIMA q", 0, 10, 0)

forecast_days = st.sidebar.slider("Forecast Days", 7, 60, 30)

# ------------------ MAIN LOGIC ------------------
if uploaded_file is None:
    st.info("👈 Upload your dataset CSV file to start.")
    st.stop()

# Read data safely (BOM-safe)
df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
df = clean_columns(df)

st.subheader("📄 Raw Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

# Detect Date + target
date_col = detect_date_column(df)
target_col = detect_target_column(df)

if target_col is None:
    st.error("❌ Target column not found. Dataset must contain Adj_Close or Close.")
    st.stop()

# Rename date column to Date
df.rename(columns={date_col: "Date"}, inplace=True)

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df = df.sort_values("Date")
df.set_index("Date", inplace=True)

# Create time series
ts = df[target_col].astype(float).dropna()

# Fix frequency for stock business days
ts = ts.asfreq("B").ffill()

st.success(f"✅ Date Column: Date | Target Column: {target_col}")
st.write(f"Total Records: **{len(ts)}**")

# ------------------ EDA SECTION ------------------
st.subheader("📊 Exploratory Data Analysis (EDA)")

col1, col2 = st.columns(2)

with col1:
    st.write("### Adj Close Trend")
    fig = plt.figure(figsize=(10, 4))
    plt.plot(ts.index, ts.values)
    plt.title("Adj Close Price Over Time")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.grid(True)
    st.pyplot(fig)

with col2:
    if "Volume" in df.columns:
        st.write("### Volume Trend")
        fig = plt.figure(figsize=(10, 4))
        plt.plot(df.index, df["Volume"])
        plt.title("Volume Over Time")
        plt.xlabel("Date")
        plt.ylabel("Volume")
        plt.grid(True)
        st.pyplot(fig)
    else:
        st.warning("⚠️ Volume column not available in dataset.")

# Returns analysis
st.write("### 📈 Daily Returns")
daily_return = ts.pct_change().dropna()

fig = plt.figure(figsize=(12, 4))
plt.plot(daily_return.index, daily_return.values)
plt.title("Daily Returns")
plt.xlabel("Date")
plt.ylabel("Return")
plt.grid(True)
st.pyplot(fig)

# ------------------ FORECAST SECTION ------------------
st.subheader("🔮 Forecasting using ARIMA")

# train-test split
split_idx = int(len(ts) * 0.8)
train, test = ts.iloc[:split_idx], ts.iloc[split_idx:]

st.write(f"Train size: **{len(train)}** | Test size: **{len(test)}**")

# Fit ARIMA
try:
    model = ARIMA(train, order=(order_p, order_d, order_q))
    fit = model.fit()
except Exception as e:
    st.error(f"❌ ARIMA Model error: {e}")
    st.stop()

# Predict on test
pred_test = fit.forecast(steps=len(test))

rmse, mae, mp = eval_metrics(test, pred_test)

metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
metrics_col1.metric("RMSE", f"{rmse:.4f}")
metrics_col2.metric("MAE", f"{mae:.4f}")
metrics_col3.metric("MAPE (%)", f"{mp:.2f}")

# Plot test forecast
st.write("### ✅ Test Forecast vs Actual")
fig = plt.figure(figsize=(12, 5))
plt.plot(test.index, test.values, label="Actual")
plt.plot(test.index, pred_test.values, label="ARIMA Forecast")
plt.title("Test Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
st.pyplot(fig)

# Forecast future
st.write(f"### 📅 Future Forecast: Next {forecast_days} Days")

future_forecast = fit.forecast(steps=forecast_days)

future_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1),
                             periods=forecast_days,
                             freq="B")

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast": future_forecast.values
})

# plot
fig = plt.figure(figsize=(12, 5))
plt.plot(ts.index[-200:], ts.values[-200:], label="Historical")
plt.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast")
plt.title("Next Days Forecast (ARIMA)")
plt.xlabel("Date")
plt.ylabel("Forecast Price")
plt.legend()
plt.grid(True)
st.pyplot(fig)

st.dataframe(forecast_df, use_container_width=True)

# Download forecast
csv = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Forecast CSV",
    data=csv,
    file_name="apple_forecast_30_days.csv",
    mime="text/csv"
)

st.success("✅ Forecast completed successfully!")
