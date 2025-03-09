import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data
def load_data():
    df = pd.read_csv("all_data1.csv", parse_dates=["order_purchase_timestamp"])
    return df


def filter_data_by_date(df, start_date, end_date):
    cols_needed = ["order_purchase_timestamp",
                   "order_id", "customer_unique_id", "payment_value"]
    df_filtered = df.loc[
        (df["order_purchase_timestamp"] >= pd.to_datetime(start_date)) &
        (df["order_purchase_timestamp"] <= pd.to_datetime(end_date)), cols_needed
    ]
    return df_filtered


def get_daily_orders(df):
    daily_orders = df.resample(rule='D', on='order_purchase_timestamp').agg({
        "order_id": "nunique",
        "payment_value": "sum"
    }).reset_index()

    daily_orders.rename(columns={
        "order_id": "order_count",
        "payment_value": "revenue"
    }, inplace=True)

    return daily_orders


def segment_customers(df):
    customer_segmentation = df.groupby("customer_unique_id").agg(
        total_spent=("payment_value", "sum")
    ).reset_index()

    customer_segmentation["spender_category"] = customer_segmentation["total_spent"].apply(
        lambda x: "High Spender" if x <= 500 else "Low Spender"
    )

    return customer_segmentation


df = load_data()

st.sidebar.header("📆 Pilih Rentang Waktu")
min_date = df["order_purchase_timestamp"].min()
max_date = df["order_purchase_timestamp"].max()

date_range = st.sidebar.date_input(
    "Pilih Rentang Waktu:", [min_date, max_date])
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

df_filtered = filter_data_by_date(df, start_date, end_date)

daily_orders = get_daily_orders(df_filtered)
customer_segmentation = segment_customers(df_filtered)

# Format Revenue
total_orders = daily_orders["order_count"].sum()
total_revenue = daily_orders["revenue"].sum()
formatted_revenue = f"AUD {total_revenue:,.0f}".replace(",", ".")

st.markdown("## Dashboard Sederhana")

col1, col2 = st.columns(2)
col1.metric("Total Orders", f"{total_orders}")
col2.metric("Total Revenue", formatted_revenue)

# Visualisasi Daily Orders
st.markdown("### Daily Orders Over Time")
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=daily_orders, x="order_purchase_timestamp", y="order_count",
             marker="o", color="blue", linewidth=1, alpha=0.7)

ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig)

df_segmented = df_filtered.merge(customer_segmentation[[
                                 "customer_unique_id", "spender_category"]], on="customer_unique_id", how="left")

daily_segmented_orders_by_category = df_segmented.groupby([pd.Grouper(key='order_purchase_timestamp', freq='D'), "spender_category"]).agg({
    "order_id": "nunique",
    "payment_value": "sum"
}).reset_index()

# --- Visualisasi Segmentasi Pelanggan ---
st.markdown("#### 📈 Daily Purchase Activity by Segment")
fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.lineplot(data=daily_segmented_orders_by_category, x="order_purchase_timestamp", y="order_id",
             hue="spender_category", style="spender_category", markers=True, dashes=False, ax=ax1)
ax1.set_xlabel("Date")
ax1.set_ylabel("Number of Orders")
ax1.set_title("Daily Purchase Activity by Customer Segment")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig1)

st.markdown("#### 💰 Daily Spending by Segment")
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.lineplot(data=daily_segmented_orders_by_category, x="order_purchase_timestamp", y="payment_value",
             hue="spender_category", style="spender_category", markers=True, dashes=False, ax=ax2)
ax2.set_xlabel("Date")
ax2.set_ylabel("Total Spending (AUD)")
ax2.set_title("Daily Spending by Customer Segment")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig2)
