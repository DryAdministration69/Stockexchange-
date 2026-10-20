import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance

st.set_page_config(
    page_title="Stock Exchange EDA & Random Forest",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Stock Exchange Data Explorer")
st.caption("Exploratory Data Analysis and Random Forest direction classification")

DATA_PATH = Path(__file__).parent / "indexData.csv"
REQUIRED_COLUMNS = {"Index", "Date", "Close"}
FEATURES = ["Previous_Close", "Daily_Return", "MA_5", "MA_10", "Price_Change"]


@st.cache_data
def load_csv(file_bytes=None):
    if file_bytes is not None:
        data = pd.read_csv(io.BytesIO(file_bytes))
    elif DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH)
    else:
        return None
    return data


def prepare_data(raw_df):
    data = raw_df.copy()
    data.columns = [str(c).strip() for c in data.columns]

    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(sorted(missing))
            + ". Expected at least Index, Date, and Close."
        )

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.drop_duplicates()
    data = data.dropna(subset=["Index", "Date", "Close"])
    data["Index"] = data["Index"].astype(str)
    data = data.sort_values(["Index", "Date"]).reset_index(drop=True)

    # Build features and next-observation target independently for each index.
    grouped = data.groupby("Index", sort=False)["Close"]
    data["Previous_Close"] = grouped.shift(1)
    data["Daily_Return"] = grouped.pct_change(fill_method=None)
    data["MA_5"] = grouped.transform(lambda s: s.rolling(5, min_periods=5).mean())
    data["MA_10"] = grouped.transform(lambda s: s.rolling(10, min_periods=10).mean())
    data["Price_Change"] = data["Close"] - data["Previous_Close"]
    data["Next_Close"] = grouped.shift(-1)
    data["Target"] = (data["Next_Close"] > data["Close"]).astype("Int64")

    data = data.replace([np.inf, -np.inf], np.nan)
    return data


def fig_to_streamlit(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# Sidebar: data input and controls
st.sidebar.header("Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file (optional)", type=["csv"])
if uploaded_file is not None:
    raw = load_csv(uploaded_file.getvalue())
else:
    raw = load_csv()

if raw is None:
    st.info(
        "No local dataset was found. Put `indexData.csv` beside `app.py`, "
        "or upload the CSV using the sidebar."
    )
    st.stop()

try:
    data = prepare_data(raw)
except Exception as exc:
    st.error(f"Could not prepare the dataset: {exc}")
    st.stop()

indices = sorted(data["Index"].dropna().unique().tolist())
selected_index = st.sidebar.selectbox("Choose stock index", ["All indices"] + indices)
if selected_index != "All indices":
    view = data[data["Index"] == selected_index].copy()
else:
    view = data.copy()

# Overview
st.header("1. Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows after cleaning", f"{len(data):,}")
c2.metric("Number of indices", f"{data['Index'].nunique():,}")
c3.metric("Date range start", data["Date"].min().strftime("%Y-%m-%d"))
c4.metric("Date range end", data["Date"].max().strftime("%Y-%m-%d"))

with st.expander("Preview data"):
    st.dataframe(view.head(100), use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Missing values in source data")
    missing_df = raw.isna().sum().rename("Missing values").to_frame()
    st.dataframe(missing_df, use_container_width=True)
with right:
    st.subheader("Descriptive statistics")
    st.dataframe(view.select_dtypes(include=np.number).describe().T, use_container_width=True)

st.header("2. Exploratory Data Analysis")
numeric_cols = view.select_dtypes(include=np.number).columns.tolist()
if numeric_cols:
    chosen_col = st.selectbox(
        "Choose a numeric column for distribution",
        [c for c in numeric_cols if c not in ["Target"]] or numeric_cols,
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(view[chosen_col].dropna(), bins=40, kde=True, ax=ax)
    ax.set_title(f"Distribution of {chosen_col}")
    fig_to_streamlit(fig)

    corr_cols = view.select_dtypes(include=np.number).columns
    corr = view[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
    ax.set_title("Numerical Feature Correlation")
    fig.tight_layout()
    fig_to_streamlit(fig)

plot_data = view.sort_values("Date")
st.subheader("Closing price over time")
fig, ax = plt.subplots(figsize=(12, 4))
if selected_index == "All indices":
    # Plot a manageable number of indices to keep the chart readable.
    top_indices = plot_data["Index"].value_counts().head(8).index
    for name, group in plot_data[plot_data["Index"].isin(top_indices)].groupby("Index"):
        ax.plot(group["Date"], group["Close"], label=name, linewidth=1)
    ax.legend(title="Index", ncol=2)
    ax.set_title("Closing prices (up to 8 indices with most records)")
else:
    ax.plot(plot_data["Date"], plot_data["Close"], linewidth=1)
    ax.set_title(f"{selected_index}: closing price over time")
ax.set_xlabel("Date")
ax.set_ylabel("Close")
fig.tight_layout()
fig_to_streamlit(fig)

if "Volume" in view.columns:
    st.subheader("Trading volume over time")
    volume_data = view.dropna(subset=["Volume"]).sort_values("Date")
    if not volume_data.empty:
        fig, ax = plt.subplots(figsize=(12, 3.5))
        if selected_index == "All indices":
            vol_group = volume_data.groupby("Date", as_index=False)["Volume"].sum()
            ax.plot(vol_group["Date"], vol_group["Volume"], linewidth=1)
            ax.set_title("Total volume across indices")
        else:
            ax.plot(volume_data["Date"], volume_data["Volume"], linewidth=1)
            ax.set_title(f"{selected_index}: trading volume")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volume")
        fig.tight_layout()
        fig_to_streamlit(fig)

st.header("3. Random Forest Model")
st.write(
    "The model predicts whether an index's next available closing price will be higher "
    "than its current close. Features are calculated separately within each index. "
    "The evaluation uses a chronological holdout rather than a random split."
)

model_data = view.dropna(subset=FEATURES + ["Target", "Next_Close"]).copy()
if selected_index == "All indices":
    st.caption("When 'All indices' is selected, rows are pooled across indices and sorted by date. "
               "For a focused analysis, select one index in the sidebar.")
else:
    st.caption(f"Model is trained and evaluated only on {selected_index}.")

if len(model_data) < 30 or model_data["Target"].nunique() < 2:
    st.warning("Not enough usable rows or target classes to train a meaningful model. Select another index or upload more data.")
    st.stop()

model_data = model_data.sort_values(["Date", "Index"]).reset_index(drop=True)
split_point = int(len(model_data) * 0.8)
train = model_data.iloc[:split_point]
test = model_data.iloc[split_point:]

if train["Target"].nunique() < 2 or test["Target"].nunique() < 2:
    st.warning("The chronological split does not contain both target classes in train and test. Try another index or dataset.")
    st.stop()

n_estimators = st.sidebar.slider("Random Forest trees", 50, 300, 100, step=50)
max_depth = st.sidebar.slider("Maximum tree depth", 3, 20, 10)

model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
model.fit(train[FEATURES], train["Target"].astype(int))
pred = model.predict(test[FEATURES])
accuracy = accuracy_score(test["Target"].astype(int), pred)

m1, m2, m3 = st.columns(3)
m1.metric("Test accuracy", f"{accuracy * 100:.2f}%")
m2.metric("Training rows", f"{len(train):,}")
m3.metric("Test rows", f"{len(test):,}")

st.subheader("Classification report")
report = classification_report(
    test["Target"].astype(int),
    pred,
    labels=[0, 1],
    target_names=["Not Up", "Up"],
    output_dict=True,
    zero_division=0,
)
st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

st.subheader("Confusion matrix")
cm = confusion_matrix(test["Target"].astype(int), pred, labels=[0, 1])
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Not Up", "Up"],
    yticklabels=["Not Up", "Up"],
    ax=ax,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
fig_to_streamlit(fig)

st.subheader("Feature importance")
importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_,
}).sort_values("Importance", ascending=False)
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=importance, x="Importance", y="Feature", ax=ax)
ax.set_title("Random Forest Feature Importance")
fig.tight_layout()
fig_to_streamlit(fig)
st.dataframe(importance, use_container_width=True)

st.subheader("Latest direction estimate")
latest = view.dropna(subset=FEATURES).sort_values("Date").tail(1)
if not latest.empty:
    latest_prediction = int(model.predict(latest[FEATURES])[0])
    probabilities = model.predict_proba(latest[FEATURES])[0]
    up_probability = float(probabilities[list(model.classes_).index(1)]) if 1 in model.classes_ else 0.0
    if latest_prediction == 1:
        st.success(f"Model estimate: next close may be UP. Estimated model probability: {up_probability:.1%}")
    else:
        st.warning(f"Model estimate: next close may NOT be UP. Estimated model probability of UP: {up_probability:.1%}")
    st.caption(f"Latest available record: {latest.iloc[0]['Index']} on {latest.iloc[0]['Date'].date()}.")
else:
    st.info("No latest row has all required features.")

st.header("4. Download results")
download_df = test[["Index", "Date", "Close", "Target"]].copy()
download_df["Predicted_Target"] = pred
download_df["Actual_Direction"] = download_df["Target"].map({0: "Not Up", 1: "Up"})
download_df["Predicted_Direction"] = download_df["Predicted_Target"].map({0: "Not Up", 1: "Up"})
st.download_button(
    "Download test predictions as CSV",
    data=download_df.to_csv(index=False).encode("utf-8"),
    file_name="stock_exchange_predictions.csv",
    mime="text/csv",
)

st.divider()
st.caption(
    "Educational demonstration only. Historical patterns and model probabilities do not guarantee future market performance. "
    "This app is not financial advice."
)
