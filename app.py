import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Apple Stock Forecast Dashboard", page_icon="🍎", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.big-title{font-size:38px;font-weight:800;color:white;margin-bottom:0px}
.sub-title{font-size:15px;color:#cfcfcf;margin-top:-5px;margin-bottom:20px}
.kpi-card{background:#111827;padding:16px;border-radius:16px;border:1px solid rgba(255,255,255,0.08);text-align:center}
.kpi-label{color:#9CA3AF;font-size:13px}
.kpi-value{color:white;font-size:24px;font-weight:800;margin-top:4px}
.kpi-small{color:#9CA3AF;font-size:12px;margin-top:4px}
</style>
""", unsafe_allow_html=True)

# ---------------- DATA PATH ----------------
DATA_PATH = "AAPL (5).csv"

# ---------------- FUNCTIONS ----------------
def clean_columns(df):
    df.columns = df.columns.astype(str).str.strip()
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True)
    df = df.loc[:, ~df.columns.duplicated()]
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

def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mp = mape(y_true, y_pred)
    return rmse, mae, mp

def add_features(df, target_col):
    df["Daily_Return"] = df[target_col].pct_change()
    df["MA20"] = df[target_col].rolling(20).mean()
    df["MA50"] = df[target_col].rolling(50).mean()
    df["MA200"] = df[target_col].rolling(200).mean()
    df["Volatility_20"] = df["Daily_Return"].rolling(20).std() * np.sqrt(252)
    df["Momentum_10"] = df[target_col] - df[target_col].shift(10)
    return df

def plot_series(x, y, title, xlabel="Date", ylabel="Value"):
    fig = plt.figure(figsize=(12,4))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    st.pyplot(fig)

def plot_multi(df, cols, title):
    fig = plt.figure(figsize=(12,5))
    for c in cols:
        plt.plot(df.index, df[c], label=c)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)

# ---------------- HEADER ----------------
st.markdown('<div class="big-title">🍎 Apple Stock Forecast Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">EDA • Technical Indicators • Forecasting (ARIMA / SARIMA / LSTM)</div>', unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
try:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
except:
    st.error(f"❌ Dataset not found: {DATA_PATH}. Put CSV in same folder as app.py")
    st.stop()

df = clean_columns(df)

date_col = detect_date_column(df)
target_col = detect_target_column(df)

if date_col is None:
    st.error("❌ Date column not found.")
    st.stop()

if target_col is None:
    st.error("❌ Target column not found (Adj_Close/Close).")
    st.stop()

df.rename(columns={date_col:"Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df.dropna(subset=[target_col])

df = df.asfreq("B")
df[target_col] = df[target_col].ffill()

df = add_features(df, target_col)

ts = df[target_col].dropna()

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")

model_choice = st.sidebar.selectbox("Select Forecast Model", ["ARIMA", "SARIMA", "LSTM", "Compare All"])

forecast_days = st.sidebar.slider("Forecast days", 7, 90, 30)

# ARIMA parameters
st.sidebar.subheader("ARIMA Parameters")
p = st.sidebar.slider("p", 0, 10, 5)
d = st.sidebar.slider("d", 0, 2, 1)
q = st.sidebar.slider("q", 0, 10, 0)

# SARIMA parameters
st.sidebar.subheader("SARIMA Parameters")
sp = st.sidebar.slider("Seasonal p", 0, 2, 1)
sd = st.sidebar.slider("Seasonal d", 0, 2, 1)
sq = st.sidebar.slider("Seasonal q", 0, 2, 1)
seasonal_period = st.sidebar.selectbox("Seasonal Period", [5, 12, 20, 30], index=2)

# LSTM params
st.sidebar.subheader("LSTM Parameters")
seq_len = st.sidebar.slider("Sequence length", 20, 120, 60)
epochs = st.sidebar.slider("Epochs", 5, 50, 10)
batch_size = st.sidebar.selectbox("Batch size", [16, 32, 64], index=1)

show_data = st.sidebar.checkbox("Show Dataset", False)

# ---------------- KPI ----------------
latest_price = ts.iloc[-1]
overall_return = ((ts.iloc[-1] - ts.iloc[0]) / ts.iloc[0]) * 100
vol = df["Volatility_20"].dropna().iloc[-1] if df["Volatility_20"].dropna().shape[0] > 0 else np.nan

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"""<div class="kpi-card"><div class="kpi-label">Latest Price</div>
<div class="kpi-value">${latest_price:,.2f}</div><div class="kpi-small">Last Business Day</div></div>""", unsafe_allow_html=True)

k2.markdown(f"""<div class="kpi-card"><div class="kpi-label">Overall Return</div>
<div class="kpi-value">{overall_return:.2f}%</div><div class="kpi-small">From first → latest</div></div>""", unsafe_allow_html=True)

k3.markdown(f"""<div class="kpi-card"><div class="kpi-label">Volatility (20D)</div>
<div class="kpi-value">{vol:.2f}</div><div class="kpi-small">Annualized</div></div>""", unsafe_allow_html=True)

k4.markdown(f"""<div class="kpi-card"><div class="kpi-label">Total Records</div>
<div class="kpi-value">{len(ts)}</div><div class="kpi-small">Business frequency</div></div>""", unsafe_allow_html=True)

st.divider()

# ---------------- RAW DATA ----------------
if show_data:
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.tail(30), use_container_width=True)

# ---------------- VISUALIZATION TABS ----------------
st.subheader("📊 Detailed Visualizations")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Price", "Moving Avg", "Returns", "Volatility", "Correlation"])

with tab1:
    plot_series(ts.index, ts.values, "Price Trend", ylabel=target_col)

with tab2:
    plot_multi(df, [target_col, "MA20", "MA50", "MA200"], "Moving Average Analysis")

with tab3:
    st.write("### Returns Over Time")
    plot_series(df.index, df["Daily_Return"].fillna(0), "Daily Returns", ylabel="Return")

    st.write("### Returns Distribution")
    fig = plt.figure(figsize=(12,4))
    plt.hist(df["Daily_Return"].dropna(), bins=60)
    plt.title("Distribution of Daily Returns")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.grid(True)
    st.pyplot(fig)

with tab4:
    plot_series(df.index, df["Volatility_20"].fillna(method="ffill"), "20-Day Rolling Volatility (Annualized)", ylabel="Volatility")

with tab5:
    st.write("### Correlation Heatmap (Numerical Features)")
    num_df = df.select_dtypes(include=[np.number]).dropna()
    corr = num_df.corr()

    fig = plt.figure(figsize=(10,6))
    plt.imshow(corr, aspect="auto")
    plt.title("Correlation Heatmap")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar()
    st.pyplot(fig)

st.divider()

# ---------------- FORECASTING ----------------
st.subheader("🔮 Forecasting Section")

# train-test split
split_idx = int(len(ts)*0.8)
train, test = ts.iloc[:split_idx], ts.iloc[split_idx:]

st.write(f"✅ Train size: {len(train)} | ✅ Test size: {len(test)}")

results_table = []

def forecast_arima():
    model = ARIMA(train, order=(p,d,q))
    fit = model.fit()
    pred_test = fit.forecast(steps=len(test))
    rmse, mae, mp = metrics(test, pred_test)

    future = fit.forecast(steps=forecast_days)
    future_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1),
                                 periods=forecast_days, freq="B")
    forecast_df = pd.DataFrame({"Date":future_dates, "Forecast":future.values})
    return pred_test, forecast_df, rmse, mae, mp

def forecast_sarima():
    model = SARIMAX(train, order=(p,d,q), seasonal_order=(sp,sd,sq,seasonal_period))
    fit = model.fit(disp=False)
    pred_test = fit.forecast(steps=len(test))
    rmse, mae, mp = metrics(test, pred_test)

    future = fit.forecast(steps=forecast_days)
    future_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1),
                                 periods=forecast_days, freq="B")
    forecast_df = pd.DataFrame({"Date":future_dates, "Forecast":future.values})
    return pred_test, forecast_df, rmse, mae, mp

def forecast_lstm():
    data = ts.values.reshape(-1,1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len,1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    pred_scaled = model.predict(X_test, verbose=0)
    pred_test = scaler.inverse_transform(pred_scaled).flatten()

    y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

    rmse, mae, mp = metrics(y_test_actual, pred_test)

    # future forecast
    last_seq = scaled[-seq_len:].reshape(1, seq_len, 1)
    future_scaled = []

    for _ in range(forecast_days):
        pred = model.predict(last_seq, verbose=0)[0][0]
        future_scaled.append(pred)
        last_seq = np.append(last_seq[:, 1:, :], [[[pred]]], axis=1)

    future = scaler.inverse_transform(np.array(future_scaled).reshape(-1,1)).flatten()
    future_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1),
                                 periods=forecast_days, freq="B")

    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future})
    pred_test_series = pd.Series(pred_test, index=test.index[-len(pred_test):])

    return pred_test_series, forecast_df, rmse, mae, mp

# --------------- RUN FORECAST ---------------
if st.button("🚀 Run Forecast"):
    if model_choice == "ARIMA":
        pred_test, forecast_df, rmse, mae, mp = forecast_arima()
        results_table.append(["ARIMA", rmse, mae, mp])

    elif model_choice == "SARIMA":
        pred_test, forecast_df, rmse, mae, mp = forecast_sarima()
        results_table.append(["SARIMA", rmse, mae, mp])

    elif model_choice == "LSTM":
        pred_test, forecast_df, rmse, mae, mp = forecast_lstm()
        results_table.append(["LSTM", rmse, mae, mp])

    else:
        pred_test_a, forecast_a, rmse_a, mae_a, mp_a = forecast_arima()
        pred_test_s, forecast_s, rmse_s, mae_s, mp_s = forecast_sarima()
        pred_test_l, forecast_l, rmse_l, mae_l, mp_l = forecast_lstm()

        results_table = [
            ["ARIMA", rmse_a, mae_a, mp_a],
            ["SARIMA", rmse_s, mae_s, mp_s],
            ["LSTM", rmse_l, mae_l, mp_l],
        ]

        # pick best by RMSE
        best = sorted(results_table, key=lambda x: x[1])[0][0]
        st.success(f"✅ Best model based on RMSE: **{best}**")

        if best == "ARIMA":
            pred_test, forecast_df = pred_test_a, forecast_a
        elif best == "SARIMA":
            pred_test, forecast_df = pred_test_s, forecast_s
        else:
            pred_test, forecast_df = pred_test_l, forecast_l

    # -------- RESULTS TABLE --------
    if len(results_table) > 0:
        st.subheader("📌 Model Evaluation Metrics")
        res_df = pd.DataFrame(results_table, columns=["Model", "RMSE", "MAE", "MAPE"])
        st.dataframe(res_df, use_container_width=True)

    # -------- PLOT TEST PRED --------
    st.subheader("✅ Forecast vs Actual (Test Set)")
    fig = plt.figure(figsize=(12,5))
    plt.plot(test.index, test.values, label="Actual")
    plt.plot(pred_test.index, pred_test.values, label="Predicted")
    plt.title("Test Forecast Comparison")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)

    # -------- FUTURE FORECAST --------
    st.subheader(f"📅 Next {forecast_days} Business Days Forecast")
    fig = plt.figure(figsize=(12,5))
    plt.plot(ts.index[-250:], ts.values[-250:], label="Historical")
    plt.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast")
    plt.title("Future Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)

    st.dataframe(forecast_df, use_container_width=True)

    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Forecast CSV", data=csv, file_name="forecast.csv", mime="text/csv")
else:
    st.info("Click **🚀 Run Forecast** to generate prediction.")
