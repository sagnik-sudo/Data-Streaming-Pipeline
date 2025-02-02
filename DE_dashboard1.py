import streamlit as st
import pandas as pd
import plotly.express as px
import time

# Set Page Layout & Dark Theme
st.set_page_config(page_title="Redset Query Dashboard", layout="wide")

# Sidebar - Navigation
st.sidebar.title("📊 Redset Query Dashboard")
page = st.sidebar.radio("Go to", ["Home", "Live Query Stream", "Trends", "Top-K Tables", "Cache Hit Rate", "Compile Time vs Joins"])

# Sidebar - Filters
st.sidebar.subheader("🔎 Filters")
cluster_id = st.sidebar.selectbox("Select Cluster ID", ["All", "Cluster 1", "Cluster 2", "Cluster 3"])
query_type = st.sidebar.selectbox("Select Query Type", ["All", "SELECT", "INSERT", "DELETE", "UPDATE"])
date_range = st.sidebar.date_input("Select Date Range", [])
user_id = st.sidebar.text_input("Enter User ID (Optional)")

# Function to load data from Parquet file
@st.cache_data
def load_data():
    return pd.read_parquet("full_sl.parquet").sample(10000, random_state=42)

df = load_data()

# KPI Metrics Section
st.markdown("## 🚀 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Queries Processed", len(df))
cache_hit_rate = (df['was_cached'].sum() / len(df)) * 100
col2.metric("Cache Hit Rate (%)", f"{cache_hit_rate:.2f}%")
avg_compile_time = df['compile_duration_ms'].mean()
col3.metric("Avg Compile Time (ms)", f"{avg_compile_time:.2f}")

# Dynamic Content Based on Selected Page
if page == "Live Query Stream":
    st.markdown("## 🔴 Live Query Stream")
    query_table = st.empty()
    while True:
        query_stream = df[['query_id', 'user_id', 'execution_duration_ms', 'was_cached']].tail(20)
        query_table.dataframe(query_stream)
        time.sleep(5)
        st.rerun()

elif page == "Top-K Tables":
    st.markdown("## 📊 Top-K Most Accessed Tables")
    
    # Convert arrival_timestamp to datetime format
    df['arrival_timestamp'] = pd.to_datetime(df['arrival_timestamp'])
    
    # Apply date filtering if selected
    if date_range:
        df = df[(df['arrival_timestamp'] >= pd.to_datetime(date_range[0])) &
                (df['arrival_timestamp'] <= pd.to_datetime(date_range[1]))]
    
    # Explode the read_table_ids column
    df_exploded = df.copy()
    df_exploded["read_table_ids"] = df_exploded["read_table_ids"].astype(str).apply(lambda x: x.split(','))
    df_exploded = df_exploded.explode("read_table_ids")
    
    # Compute Top-K Tables
    top_tables = df_exploded["read_table_ids"].value_counts().reset_index()
    top_tables.columns = ["Table ID", "Query Count"]
    top_tables["Percentage"] = (top_tables["Query Count"] / top_tables["Query Count"].sum()) * 100
    
    # Plot the Top-K Chart
    if not top_tables.empty:
        fig = px.bar(top_tables.head(10), x="Table ID", y="Query Count", color="Query Count", 
                     color_continuous_scale="Blues", title="🔝 Top-K Most Accessed Tables")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")

elif page == "Cache Hit Rate":
    st.markdown("## ⚡ Cache Hit Rate Over Time")
    st.write("(Graph will be implemented here)")

elif page == "Compile Time vs Joins":
    st.markdown("## ⏳ Compile Time vs Number of Joins")
    st.write("(Graph will be implemented here)")


