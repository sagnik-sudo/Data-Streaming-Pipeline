import streamlit as st
import asyncio
import asyncpg
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from collections import Counter
import os
import hashlib
import pickle
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import numpy as np
from datetime import date, timedelta
import time


# Set Page Layout & Dark Theme
st.set_page_config(page_title="TJW Dash", layout="wide", menu_items=None)

# -------------------------------
# 🔹 USER AUTHENTICATION SYSTEM 🔹
# -------------------------------
CREDENTIALS_FILE = "bin/user_credentials.bin"
ADMIN_USER = "admin"  # Change this to your desired admin username

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to save user credentials
def save_credentials(username, password):
    credentials = {}

    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, "rb") as f:
            try:
                credentials = pickle.load(f)
            except EOFError:
                credentials = {}

    credentials[username] = hash_password(password)

    with open(CREDENTIALS_FILE, "wb") as f:
        pickle.dump(credentials, f)

# Function to validate user credentials
def validate_credentials(username, password):
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, "rb") as f:
            try:
                credentials = pickle.load(f)
                return credentials.get(username) == hash_password(password)
            except EOFError:
                return False
    return False

# Function to load all registered users
def get_registered_users():
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, "rb") as f:
            try:
                credentials = pickle.load(f)
                return list(credentials.keys())  # Return usernames only (not passwords)
            except EOFError:
                return []
    return []

# Function to update username
def update_username(old_username, new_username):
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, "rb") as f:
            credentials = pickle.load(f)

        if new_username in credentials:
            return False  # Username already exists

        credentials[new_username] = credentials.pop(old_username)

        with open(CREDENTIALS_FILE, "wb") as f:
            pickle.dump(credentials, f)
        return True
    return False

# Function to update user password
def update_password(username, new_password):
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, "rb") as f:
            credentials = pickle.load(f)

        credentials[username] = hash_password(new_password)

        with open(CREDENTIALS_FILE, "wb") as f:
            pickle.dump(credentials, f)
        return True
    return False

# Function to delete a user
def delete_user(username):
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, "rb") as f:
            credentials = pickle.load(f)

        if username in credentials:
            del credentials[username]

            with open(CREDENTIALS_FILE, "wb") as f:
                pickle.dump(credentials, f)
            return True
    return False

# Ensure session state is initialized
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "current_page" not in st.session_state:
    st.session_state.current_page = "Live Insights Panel"  # Default to Page 1

# -------------------------------------------
# 🔹 LOGIN & REGISTRATION PAGE
# -------------------------------------------
if not st.session_state.logged_in:
    st.title("TJW Redset Dashboard")
    menu = st.sidebar.radio("Choose Action", ["Login", "Register"])

    if menu == "Register":
        st.subheader("📌 Register New User")
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        if st.button("Register"):
            if new_username and new_password:
                save_credentials(new_username, new_password)
                st.success(f"✅ User '{new_username}' registered successfully! You can now log in.")
            else:
                st.warning("⚠️ Please enter both username and password.")

    elif menu == "Login":
        st.subheader("🔑 User Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            if validate_credentials(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("❌ Invalid username or password. Try again.")

    st.stop()

# -------------------------------------------
# 🔹 DASHBOARDS (Only Visible After Login)
# -------------------------------------------
def linear(x, a, b):
    return a * x + b

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def exponential(x, a, b):
    return a * np.exp(b * x)

# Log transformation function
def log_transform(y):
    return np.log1p(y)  # log(y + 1) to avoid log(0) errors

# Inverse transformation (for plotting)
def inverse_log_transform(y_log):
    return np.expm1(y_log)  # exp(y) - 1

# Database Configuration
# DB_CONFIG = {
#     'user': 'sagnik',
#     'password': 'sagnik',
#     'host': '192.168.127.138',
#     'port': '5432',
#     'database': 'de_project_main'
# }
DB_CONFIG = {
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432',
    'database': 'redset'
}

# Function to fetch PostgreSQL data
def run_async_query(query: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(fetch_data(query))

async def fetch_data(query: str):
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        records = await conn.fetch(query)
        await conn.close()
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records, columns=records[0].keys())
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()


# -------------------------------------------
# 🔹 ADMIN DASHBOARD
# -------------------------------------------
if st.session_state.username == ADMIN_USER:
    st.title("🎛️ Command Centre")
    st.subheader("👥 Registered Users")

    registered_users = get_registered_users()
    if registered_users:
        df_users = pd.DataFrame({"Username": registered_users})
        name_mapping = {
            "sagnik": "Sagnik Das",
            "admin": "Dashboard Administrator",
            "aswathy": "Aswathy Marottikal Ramesh",
            "goutham": "Goutham Muralikrishnan",
            "muhid": "Muhid Abid Siddiqui",
            "pranjaly": "Pranjaly Paul"
        }

        df_users["Full Name"] = df_users["Username"].map(name_mapping).fillna("")
        df_users.index = df_users.index + 1
        st.dataframe(df_users)

        selected_user = st.selectbox("Select a user:", registered_users)

        if selected_user != ADMIN_USER:
            new_username = st.text_input("New Username", value=selected_user)
            if st.button("Update Username"):
                if update_username(selected_user, new_username):
                    st.success(f"✅ Username changed to '{new_username}'. Please refresh.")
                else:
                    st.error("❌ Username already exists.")

            new_password = st.text_input("New Password", type="password")
            if st.button("Update Password"):
                if new_password:
                    update_password(selected_user, new_password)
                    st.success("✅ Password updated successfully.")
                else:
                    st.warning("⚠️ Enter a valid password.")

            if st.button("Delete User"):
                if delete_user(selected_user):
                    st.success(f"✅ User '{selected_user}' deleted.")
                    st.rerun()
                else:
                    st.error("❌ Unable to delete user.")

# Sidebar - Navigation

st.sidebar.title(f"👋 Welcome, {st.session_state.username.capitalize()}")
# st.sidebar.title("📊 TJW Redset Dashboard")
page = st.sidebar.radio("Go to", ["Home", "Live Query Stream", "Trends", "Top-K Tables", "Cache Hit Rate", "Compile Time vs Joins"])

# Sidebar - Filters
st.sidebar.subheader("🔎 Filters")
# cluster_id = st.sidebar.selectbox("Select Cluster ID", ["All", "Cluster 1", "Cluster 2", "Cluster 3"])
# Set default start date to 2024-01-01
default_start = date(2024, 1, 1)
# Set default end date to 90 days after 2024-01-01
default_end = default_start + timedelta(days=90)
date_range = st.sidebar.date_input("Select Date Range", [default_start, default_end])

# Ensure exactly two dates are selected
if len(date_range) == 2:
    start_date, end_date = date_range
else:
    st.error("⚠️ Please select both a start and an end date.")
    st.stop()  # Stop execution until the user provides both dates

# Number input for User ID (default is None)
user_id = st.sidebar.number_input("Enter User ID (Optional)", min_value=1, step=1, format="%d", value=None)

# Convert None to empty string for handling cases
user_id = int(user_id) if user_id else None

query_type_options = ["(Select Query Type)", "delete", "update", "unload", "vacuum", "analyze", "other", "select", "copy", "ctas", "insert"]

# Dropdown with optional selection
query_type = st.sidebar.selectbox("Select Query Type (Optional)", query_type_options)

# Treat the default "(Select Query Type)" as None
query_type = None if query_type == "(Select Query Type)" else query_type

# Dynamic Content Based on Selected Page
if page == "Home":
    # KPI Metrics Section
    st.markdown("## 🚀 Key Metrics")
    col1, col2, col3 = st.columns(3)
    total_queries = run_async_query("SELECT COUNT(*) FROM redset_main")
    total_queries_count = total_queries.iloc[0, 0]
    col1.metric("Total Queries Processed", total_queries_count)
    # Execute SQL query
    cache_hit_rate_query = run_async_query("SELECT (SUM(was_cached) * 100.0) / COUNT(*) AS cache_hit_rate FROM redset_main")

    # Extract value from the result
    cache_hit_rate = cache_hit_rate_query.iloc[0, 0]  # Assuming it returns a DataFrame

    # Display in Streamlit
    col2.metric("Cache Hit Rate (%)", f"{cache_hit_rate:.2f}")
    # Run SQL query to get the average compile time
    avg_compile_time_query = run_async_query("SELECT COALESCE(compile_duration_ms, 0) AS avg_compile_time FROM redset_main")

    # Extract value from the result
    avg_compile_time = avg_compile_time_query.iloc[0, 0]  # Assuming it returns a DataFrame

    # Display in Streamlit
    col3.metric("Average Compile Time (ms)", f"{avg_compile_time:.2f}")

elif page == "Trends":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4>📊 Was Cached Distribution</h4>", unsafe_allow_html=True)
        df = run_async_query("SELECT * FROM redset_main")
        cached_count = df["was_cached"].value_counts().reset_index()
        cached_count.columns = ["was_cached", "count"]
        fig = px.bar(cached_count, x="was_cached", y="count", title="Query Count by 'was_cached'")
        st.plotly_chart(fig)
    with col2:
        st.markdown("<h4>📊 Query Compilation vs Execution Time", unsafe_allow_html=True)
        # Fetch only necessary columns
        query = """
            SELECT arrival_timestamp, compile_duration_ms, execution_duration_ms 
            FROM redset_main 
            WHERE arrival_timestamp IS NOT NULL
        """
        df = run_async_query(query)

        # Ensure necessary columns exist
        if {"arrival_timestamp", "compile_duration_ms", "execution_duration_ms"}.issubset(df.columns):
            # Convert timestamp to datetime
            df["arrival_timestamp"] = pd.to_datetime(df["arrival_timestamp"])

            # Group by hour and compute average durations
            df["hour"] = df["arrival_timestamp"].dt.floor("H")  # Rounds to nearest hour
            hourly_avg = df.groupby("hour").agg({
                "compile_duration_ms": "mean",
                "execution_duration_ms": "mean"
            }).reset_index()

            # Create a line chart
            fig = px.line(
                hourly_avg, 
                x="hour", 
                y=["compile_duration_ms", "execution_duration_ms"], 
                labels={"value": "Average Duration (ms)", "hour": "Time (Hourly)"},
                title="Average Query Compilation vs Execution Time Per Hour"
            )

            st.plotly_chart(fig)

        else:
            st.warning("Required columns not found in the dataset. Please check the data schema.")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<h4>📊 Cached vs Non-Cached Query Execution Time</h4>", unsafe_allow_html=True)

        # Fetch only necessary columns
        query = """
            SELECT query_type, was_cached, AVG(execution_duration_ms) AS avg_execution_time_ms
            FROM redset_main
            WHERE query_type IS NOT NULL AND was_cached IS NOT NULL
            GROUP BY query_type, was_cached
            ORDER BY avg_execution_time_ms ASC
        """
        df = run_async_query(query)

        # Ensure necessary columns exist
        if {"query_type", "was_cached", "avg_execution_time_ms"}.issubset(df.columns):
            # Convert was_cached to string for better labels
            df["was_cached"] = df["was_cached"].astype(str).replace({"0": "Not Cached", "1": "Cached"})

            # Create bar chart
            fig = px.bar(
                df, 
                x="query_type", 
                y="avg_execution_time_ms", 
                color="was_cached",  # Color by cached vs non-cached
                barmode="group",  # Group bars side-by-side
                log_y=True,
                labels={"query_type": "Query Type", "avg_execution_time_ms": "Avg Execution Time (ms)"},
                title="Execution Time Comparison: Cached vs Non-Cached Queries"
            )

            st.plotly_chart(fig)

        else:
            st.warning("Required columns not found in the dataset. Please check the data schema.")
    with col4:
        st.markdown("<h4>📊 Join vs Scan Efficiency</h4>", unsafe_allow_html=True)

        # Fetch only necessary columns
        query = """
            SELECT query_type, 
                AVG(num_joins) AS avg_joins, 
                AVG(num_scans) AS avg_scans 
            FROM redset_main
            WHERE query_type IS NOT NULL
            GROUP BY query_type
            ORDER BY avg_scans DESC, avg_joins DESC
        """
        df = run_async_query(query)

        # Ensure necessary columns exist
        if {"query_type", "avg_joins", "avg_scans"}.issubset(df.columns):
            # Melt dataframe for better visualization
            df_melted = df.melt(
                id_vars=["query_type"], 
                value_vars=["avg_joins", "avg_scans"], 
                var_name="Metric", 
                value_name="Average Count"
            )

            # Create a grouped bar chart
            fig = px.bar(
                df_melted, 
                x="query_type", 
                y="Average Count", 
                color="Metric",  # Color by joins vs scans
                barmode="group",  # Group bars side-by-side
                labels={"query_type": "Query Type", "Average Count": "Avg Joins & Scans"},
                title="Join vs Scan Efficiency by Query Type"
            )

            st.plotly_chart(fig)

        else:
            st.warning("Required columns not found in the dataset. Please check the data schema.")

elif page == "Live Query Stream":
    # st.title("📊 Live Query Stream")

    # if start_date and end_date:
    #     start_date_str = start_date.strftime('%Y-%m-%d')
    #     end_date_str = end_date.strftime('%Y-%m-%d')

    #     query = f"""
    #     SELECT 
    #         date_trunc('day', rm.arrival_timestamp) AS day,
    #         table_id,
    #         rm.query_type,
    #         rm.user_id,
    #         COUNT(*) AS count,
    #         COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY date_trunc('day', rm.arrival_timestamp)) AS percentage,
    #         COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS overall_percentage
    #     FROM (
    #         SELECT 
    #             arrival_timestamp, 
    #             unnest(string_to_array(read_table_ids, ',')) AS table_id,
    #             query_type,
    #             user_id
    #         FROM redset_main
    #         WHERE arrival_timestamp BETWEEN '{start_date_str}' AND '{end_date_str}'
    #         UNION ALL
    #         SELECT 
    #             arrival_timestamp, 
    #             unnest(string_to_array(write_table_ids, ',')) AS table_id,
    #             query_type,
    #             user_id
    #         FROM redset_main
    #         WHERE arrival_timestamp BETWEEN '{start_date_str}' AND '{end_date_str}'
    #     ) AS rm
    #     GROUP BY day, table_id, rm.query_type, rm.user_id
    #     ORDER BY count DESC
    #     LIMIT 50;
    #     """
    # else:
    #     query = "SELECT * FROM public.top_k_tables_per_day LIMIT 50"

    # # **Live Data Streaming Section**
    # placeholder = st.empty()  # Placeholder for real-time updates

    # # Refresh Rate Selection
    # refresh_rate = st.slider("Refresh Interval (seconds)", 1, 30, 10)
    # top_k = st.number_input("Select the number of top tables to view", min_value=1, max_value=50, value=10)
    # while True:
    #     df = run_async_query(query)

    #     if not df.empty:
    #         df_grouped = df.groupby('table_id')[['count', 'percentage']].sum().reset_index()
    #         df_grouped = df_grouped.sort_values(by='count', ascending=False)

    #         # Calculate overall percentage for the date range
    #         total_count = df_grouped['count'].sum()
    #         df_grouped['overall_percentage'] = (df_grouped['count'] / total_count) * 100

    #         # Limit to top_k tables
    #         df_grouped = df_grouped.head(top_k)

    #         # Plot Data
    #         fig, ax = plt.subplots(figsize=(10, 6))
    #         bars = ax.bar(df_grouped['table_id'], df_grouped['count'])

    #         ax.set_xlabel('Table ID')
    #         ax.set_ylabel('Query Count')
    #         ax.set_title('Top K Tables')
    #         ax.set_xticklabels(df_grouped['table_id'], rotation=90)

    #         # Display percentage on top of bars diagonally
    #         for bar, percentage in zip(bars, df_grouped['overall_percentage']):
    #             ax.text(
    #                 bar.get_x() + bar.get_width() / 2,
    #                 bar.get_height(),
    #                 f'{percentage:.2f}%',
    #                 ha='center',
    #                 va='bottom',
    #                 rotation=45
    #             )

    #         # Update the Streamlit UI dynamically
    #         with placeholder:
    #             st.pyplot(fig)
    #             st.write(f"🔄 Last updated: {time.strftime('%H:%M:%S')}")

    #     else:
    #         with placeholder:
    #             st.warning("⚠️ No data available for the selected date range.")

    #     time.sleep(refresh_rate)  # Wait for the specified refresh interval
    graph2_placeholder = st.empty()
    graph3_placeholder = st.empty()

    # Refresh Rate Selector
    refresh_rate = st.slider("Refresh Interval (seconds)", 1, 30, 5)
    while True:
        query_compile_time_vs_joins = f"""
        SELECT num_joins AS x, AVG(compile_duration_ms) AS y
        FROM public.redset_main
        WHERE query_type = 'select' 
        AND num_joins IS NOT NULL
        AND arrival_timestamp BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY num_joins
        ORDER BY num_joins;
        """

        # Fetch Data
        
        df = run_async_query(query_compile_time_vs_joins)

        if not df.empty:
            # Convert to numerical values for plotting
            df["x"] = pd.to_numeric(df["x"])
            df["y"] = pd.to_numeric(df["y"])

            # Plot Data
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(df["x"], df["y"], color="blue", alpha=0.6, label="Data Points")

            ax.set_xlabel("Number of Joins")
            ax.set_ylabel("Compile Duration (ms)")
            ax.set_title("Compile Time vs Number of Joins")
            ax.grid(True)
            ax.legend()

            # Display plot in Streamlit
            graph2_placeholder.pyplot(fig)
            # plt.close(fig)
            # st.pyplot(fig)
        else:
            st.warning("⚠️ No data available for Compile Time vs Number of Joins.")
        if start_date and end_date:
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            query = f"""
                SELECT 
                    date_trunc('day', arrival_timestamp) AS day,
                    SUM(COUNT(*) FILTER (WHERE was_cached = 1)) OVER (PARTITION BY date_trunc('day', arrival_timestamp)) * 100.0 
                    / NULLIF(SUM(COUNT(*)) OVER (PARTITION BY date_trunc('day', arrival_timestamp)), 0) AS total_hit_rate_per_day
                FROM redset_main
                WHERE arrival_timestamp BETWEEN '{start_date_str}' AND '{end_date_str}'
                GROUP BY day
                ORDER BY day;
            """
        else:
            query = """
                SELECT 
                    date_trunc('day', arrival_timestamp) AS day,
                    SUM(COUNT(*) FILTER (WHERE was_cached = 1)) OVER (PARTITION BY date_trunc('day', arrival_timestamp)) * 100.0 
                    / NULLIF(SUM(COUNT(*)) OVER (PARTITION BY date_trunc('day', arrival_timestamp)), 0) AS total_hit_rate_per_day
                FROM redset_main
                GROUP BY day
                ORDER BY day;
            """

        # Fetch Data
        df = run_async_query(query)

        if not df.empty:
            # Convert "day" to string (for cleaner x-axis labels)
            df["day"] = df["day"].astype(str)

            # Plot Data
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df["day"], df["total_hit_rate_per_day"], marker="o", linestyle="-", color="blue", label="Cache Hit Rate")
            ax.set_xlabel("Date")
            ax.set_ylabel("Cache Hit Rate (%)")
            ax.set_title("Cache Hit Rate Over Time")
            ax.grid(True)

            # Set only dates (remove time component)
            ax.set_xticks(df["day"])  # Set the ticks to be the date values
            ax.set_xticklabels(df["day"], rotation=45)  # Rotate for better readability
            
            ax.legend()

            # Display plot in Streamlit
            graph2_placeholder.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("⚠️ No data available for the selected date range.")

elif page == "Top-K Tables":
    st.title("📊 Top K Tables")

    if start_date and end_date:
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        query = f"""
        SELECT * FROM public.top_k_tables_per_day 
        WHERE arrival_timestamp BETWEEN '{start_date_str}' AND '{end_date_str}';
        """
    else:
        query = "SELECT * FROM public.top_k_tables_per_day"

    # Fetch Data
    df = run_async_query(query)

    if not df.empty:
        # Let the user select how many top tables to display
        top_k = st.number_input("Select the number of top tables to view", min_value=1, max_value=50, value=10)

        df_grouped = df.groupby('table_id')[['count', 'percentage']].sum().reset_index()
        df_grouped = df_grouped.sort_values(by='count', ascending=False)

        # Calculate overall percentage for the date range
        total_count = df_grouped['count'].sum()
        df_grouped['overall_percentage'] = (df_grouped['count'] / total_count) * 100

        # Limit to top_k tables
        df_grouped = df_grouped.head(top_k)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(df_grouped['table_id'], df_grouped['count'])
        plt.xlabel('Table ID')
        plt.ylabel('Query Count')
        plt.title('Top K Tables')
        plt.xticks(rotation=90)

        # Display percentage on top of bars diagonally
        for bar, percentage in zip(bars, df_grouped['overall_percentage']):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f'{percentage:.2f}%',
                ha='center',
                va='bottom',
                rotation=45
            )

        st.pyplot(plt)

    else:
        st.write("⚠️ No data available for the selected date range.")


elif page == "Cache Hit Rate":
    st.markdown("## ⚡ Cache Hit Rate Over Time Range")
    if start_date and end_date:
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        query = f"""
            SELECT day::date AS day, SUM(hit_rate_per_day) AS cache_hit_rate
            FROM public.hit_rate_per_day
            WHERE day BETWEEN '{start_date_str}' AND '{end_date_str}'
            GROUP BY day
            ORDER BY day ASC;
        """
    else:
        query = """
            SELECT day::date AS day, SUM(hit_rate_per_day) AS cache_hit_rate
            FROM public.hit_rate_per_day
            GROUP BY day
            ORDER BY day ASC;
        """

    # Fetch Data
    df = run_async_query(query)

    if not df.empty:
        # Convert "day" to string (for cleaner x-axis labels)
        df["day"] = df["day"].astype(str)

        # Plot Data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["day"], df["cache_hit_rate"], marker="o", linestyle="-", color="blue")

        ax.set_xlabel("Date")
        ax.set_ylabel("Cache Hit Rate (%)")
        # ax.set_title("Cache Hit Rate Over Time")
        ax.grid(True)

        # Set only dates (remove time component)
        ax.set_xticks(df["day"])  # Set the ticks to be the date values
        ax.set_xticklabels(df["day"], rotation=45)  # Rotate for better readability
        
        # Display plot in Streamlit
        st.pyplot(fig)
    else:
        st.warning("⚠️ No data available for the selected date range.")

elif page == "Compile Time vs Joins":
    # -------------------------------------------
    # 🔹 Compile Time vs Number of Joins (With Log Transformation)
    # -------------------------------------------
    st.title("⏳ Compile Time vs Number of Joins")

    # Fetch data for the visualization
    query_compile_time_vs_joins = """
        SELECT x, y
        FROM public.compile_time_vs_num_joins;
    """
    df_compile_time_vs_joins = run_async_query(query_compile_time_vs_joins)

    if not df_compile_time_vs_joins.empty:
        x = df_compile_time_vs_joins["x"].values
        y = df_compile_time_vs_joins["y"].values

        # Apply log transformation to y
        y_log = log_transform(y)

        # Fit models on transformed data
        try:
            popt_linear, _ = curve_fit(linear, x, y_log)
            y_pred_linear = inverse_log_transform(linear(x, *popt_linear))
            r2_linear = r2_score(y, y_pred_linear)
        except:
            r2_linear = -np.inf

        try:
            popt_quadratic, _ = curve_fit(quadratic, x, y_log)
            y_pred_quadratic = inverse_log_transform(quadratic(x, *popt_quadratic))
            r2_quadratic = r2_score(y, y_pred_quadratic)
        except:
            r2_quadratic = -np.inf

        try:
            popt_exponential, _ = curve_fit(exponential, x, y_log, maxfev=10000)
            y_pred_exponential = inverse_log_transform(exponential(x, *popt_exponential))
            r2_exponential = r2_score(y, y_pred_exponential)
        except:
            r2_exponential = -np.inf

        # Select the best-fitting model
        best_fit = max(
            [("Linear", r2_linear, y_pred_linear if r2_linear != -np.inf else None), 
            ("Quadratic", r2_quadratic, y_pred_quadratic if r2_quadratic != -np.inf else None), 
            ("Exponential", r2_exponential, y_pred_exponential if r2_exponential != -np.inf else None)], 
            key=lambda x: x[1]
        )
        
        best_model, best_r2, best_y_pred = best_fit

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x, y, label="Data Points", color="blue", alpha=0.6)

        if best_y_pred is not None:
            ax.plot(x, best_y_pred, label=f"Best Fit: {best_model} (R²={best_r2:.3f})", color="red", linewidth=2)

        ax.set_xlabel("Number of Joins")
        ax.set_ylabel("Compile Duration (ms)")
        # ax.set_title("Compile Time vs Number of Joins")
        ax.legend()
        
        st.pyplot(fig)

        st.write(f"📈 Best Fit Model: **{best_model}** with R² = **{best_r2:.3f}**")
        
    else:
        st.write("⚠️ No data available for Compile Time vs Number of Joins.")




# Logout Button
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()
