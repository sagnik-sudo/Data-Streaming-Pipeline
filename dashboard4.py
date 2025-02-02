import streamlit as st         # For building the dashboard UI
import pandas as pd            # For data manipulation
import plotly.express as px    # For creating interactive plots
import datetime               # For handling dates
import numpy as np            # For simulating live data changes (optional)

# Import the autorefresh component.
# Make sure to install it: pip install streamlit-autorefresh
from streamlit_autorefresh import st_autorefresh

# -----------------------------------------------------------------------------
# SET UP THE STREAMLIT DASHBOARD
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Redset Query Dashboard", layout="wide")

# -----------------------------
# Sidebar: Navigation & Filters
# -----------------------------
st.sidebar.title("📊 Redset Query Dashboard")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Live Query Stream", "Trends", "Top-K Tables", "Cache Hit Rate", "Compile Time vs Joins"]
)

st.sidebar.subheader("🔎 Filters")
cluster_id = st.sidebar.selectbox("Select Cluster ID", ["All", "Cluster 1", "Cluster 2", "Cluster 3"])
query_type = st.sidebar.selectbox("Select Query Type", ["All", "SELECT", "INSERT", "DELETE", "UPDATE"])

# Default date range based on your dataset dates
default_start = datetime.date(2024, 3, 1)
default_end = datetime.date(2024, 5, 31)
date_range = st.sidebar.date_input("Select Date Range", [default_start, default_end])

# Optional user filter
user_id = st.sidebar.text_input("Enter User ID (Optional)")

# -----------------------------------------------------------------------------
# FUNCTION: Load Data (with caching for better performance)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Replace "cleaned_data.parquet" with your actual dataset file name.
    df = pd.read_parquet("cleaned_data.parquet").sample(10000, random_state=42)
    # Ensure the arrival_timestamp column is in datetime format.
    df['arrival_timestamp'] = pd.to_datetime(df['arrival_timestamp'])
    return df

df = load_data()

# -----------------------------------------------------------------------------
# APPLY DATE RANGE FILTERING
# -----------------------------------------------------------------------------
if len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df[
        (df['arrival_timestamp'] >= pd.to_datetime(start_date)) &
        (df['arrival_timestamp'] <= pd.to_datetime(end_date))
    ]
else:
    df_filtered = df

# -----------------------------------------------------------------------------
# HOME PAGE: Engaging, Colorful, and Informative
# -----------------------------------------------------------------------------
if page == "Home":
    st.markdown("# 🎉 Welcome to the Redset Query Dashboard!")
    st.write("A powerful visualization tool for monitoring query performance and cache efficiency.")

    # 🌟 Key Performance Indicators (KPIs)
    st.markdown("## 🚀 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)  # Divide into four columns
    col1.metric("Total Queries Processed", len(df_filtered))  # Display total queries
    cache_hit_rate = (df_filtered['was_cached'].sum() / len(df_filtered)) * 100
    col2.metric("Cache Hit Rate (%)", f"{cache_hit_rate:.2f}%")  # Cache hit rate
    avg_compile_time = df_filtered['compile_duration_ms'].mean()
    col3.metric("Avg Compile Time (ms)", f"{avg_compile_time:.2f}")  # Average compilation time
    most_frequent_query = df_filtered['query_type'].mode()[0]
    col4.metric("Most Frequent Query Type", most_frequent_query)  # Most common query type

    # 📈 Arrange graphs in a grid layout for better visibility
    st.markdown("## 📊 Visual Analytics")
    col1, col2 = st.columns(2)  # Split into two columns for better fitting

    with col1:
        st.markdown("### Query Execution Rate Over Time")
        df_temp = df_filtered.groupby(pd.Grouper(key='arrival_timestamp', freq='1H'))['query_id'].count().reset_index()
        fig = px.line(df_temp, x='arrival_timestamp', y='query_id', markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 🍩 Query Type Distribution (Donut Chart)
        st.markdown("## 🍩 Query Type Distribution")
        query_distribution = df_filtered['query_type'].value_counts().reset_index()
        query_distribution.columns = ['Query Type', 'Count']
        fig_donut = px.pie(query_distribution, names='Query Type', values='Count', hole=0.4,
                       title="Query Type Breakdown")
        st.plotly_chart(fig_donut, use_container_width=True)

    
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Memory Usage Trends (7-Day Rolling Average)")
        mem_trend = df_filtered.groupby(pd.Grouper(key='arrival_timestamp', freq='1D'))[['mbytes_scanned', 'mbytes_spilled']].sum().reset_index()
        mem_trend['mbytes_scanned'] = mem_trend['mbytes_scanned'].rolling(window=7, min_periods=1).mean()
        mem_trend['mbytes_spilled'] = mem_trend['mbytes_spilled'].rolling(window=7, min_periods=1).mean()
        fig_mem = px.line(mem_trend, x='arrival_timestamp', y=['mbytes_scanned', 'mbytes_spilled'], markers=True)
        st.plotly_chart(fig_mem, use_container_width=True)

    with col4:
        st.markdown("## 📜 Query Execution Summary")
        st.dataframe(df_filtered[['query_id', 'query_type', 'execution_duration_ms', 'was_cached']].tail(10))
        

    # 📜 Auto-Updating Summary Table
    #st.markdown("## 📜 Query Execution Summary")
    #st.dataframe(df_filtered[['query_id', 'query_type', 'execution_duration_ms', 'was_cached']].tail(10))

# Adjust plot sizes for better visualization without scrolling
    st.markdown("<style>div.Widget.row-widget.stPlotlyChart {height: 300px !important;}</style>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# LIVE QUERY STREAM PAGE: Auto-Refreshing Top-K Most Accessed Tables Histogram
# -----------------------------------------------------------------------------
elif page == "Live Query Stream":
    st.markdown("## 🔴 Live Query Stream - Top-K Most Accessed Tables Histogram")
    
    # Auto-refresh the entire page every 1 second (only on this page) 
    st_autorefresh(interval=1000, limit=0, key="autorefresh_live")
    
    # Process the data for the histogram:
    df_temp = df_filtered.copy()
    # Convert 'read_table_ids' to a list if it is a string.
    df_temp["read_table_ids"] = df_temp["read_table_ids"].apply(
        lambda x: x.split(',') if isinstance(x, str) else x
    )
    # Explode the list so that each table ID appears on its own row.
    df_exploded = df_temp.explode("read_table_ids")
    # Group by table ID and count the number of unique queries.
    top_tables = df_exploded.groupby("read_table_ids")["query_id"].nunique().reset_index()
    top_tables.columns = ["Table ID", "Query Count"]
    # Calculate the percentage of queries that include each table.
    total_queries = df_temp["query_id"].nunique()
    top_tables["Percentage"] = (top_tables["Query Count"] / total_queries) * 100

    # (Optional) Simulate live data changes by adding random noise.
    # Uncomment the following lines to simulate changes.
    # noise = np.random.randint(-2, 3, size=top_tables.shape[0])
    # top_tables["Query Count"] = (top_tables["Query Count"] + noise).clip(lower=0)
    # top_tables["Percentage"] = (top_tables["Query Count"] / total_queries) * 100

    # Select the top 10 tables.
    top_k_tables = top_tables.sort_values("Query Count", ascending=False).head(10)
    
    # Create the histogram.
    fig = px.bar(
        top_k_tables,
        x="Table ID",
        y="Percentage",
        title="Top-K Most Accessed Tables (%)",
        labels={"Percentage": "% of Queries"},
        color="Percentage",             # Color the bars by percentage (optional)
        color_continuous_scale="Blues"    # Use a blue color scale (optional)
    )
    
    # Display the chart with a stable key.
    st.plotly_chart(fig, use_container_width=True, key="live_chart")

# -----------------------------------------------------------------------------
# TRENDS PAGE (Placeholder)
# -----------------------------------------------------------------------------
elif page == "Trends":
    st.markdown("## 📈 Trends")
    st.write("Trends page coming soon!")

# -----------------------------------------------------------------------------
# TOP-K TABLES PAGE (Placeholder)
# -----------------------------------------------------------------------------
elif page == "Top-K Tables":
    st.markdown("## 🔝 Top-K Tables")
    st.write("Top-K Tables page coming soon!")

# -----------------------------------------------------------------------------
# CACHE HIT RATE OVER TIME PAGE
# -----------------------------------------------------------------------------
elif page == "Cache Hit Rate":
    st.markdown("## ⚡ Cache Hit Rate Over Time")
    if not df_filtered.empty and "was_cached" in df_filtered.columns:
        # Resample data by day and aggregate cache hits and total queries.
        cache_data = df_filtered.resample("1D", on="arrival_timestamp").agg(
            {"was_cached": "sum", "query_id": "count"}
        ).reset_index()
        cache_data["cache_hit_rate"] = (cache_data["was_cached"] / cache_data["query_id"]) * 100
        
        # Create a line chart for cache hit rate over time.
        fig = px.line(
            cache_data,
            x="arrival_timestamp",
            y="cache_hit_rate",
            title="Cache Hit Rate Over Time",
            labels={"cache_hit_rate": "Cache Hit Rate (%)", "arrival_timestamp": "Date"},
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Cache hit data is not available for the selected date range.")

# -----------------------------------------------------------------------------
# COMPILE TIME VS JOINS PAGE
# -----------------------------------------------------------------------------
elif page == "Compile Time vs Joins":
    st.markdown("## ⏳ Compile Time vs Number of Joins")
    if not df_filtered.empty and "num_joins" in df_filtered.columns and "compile_duration_ms" in df_filtered.columns:
        if "query_type" in df_filtered.columns:
            select_queries = df_filtered[df_filtered["query_type"] == "SELECT"]
        else:
            select_queries = df_filtered
        if not select_queries.empty:
            fig = px.scatter(
                select_queries,
                x="num_joins",
                y="compile_duration_ms",
                title="Compile Time vs Number of Joins",
                trendline="ols",
                color="num_joins"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No SELECT queries available for this filter.")
    else:
        st.warning("Required data is not available for the selected date range.")

# -----------------------------------------------------------------------------
# DEFAULT / HOME (Fallback)
# -----------------------------------------------------------------------------
else:
    st.markdown("## Welcome to the Redset Query Dashboard")
    st.write("Use the sidebar to navigate to different pages.")

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.write("Dashboard Developed with ❤️ using Streamlit")
