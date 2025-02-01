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
        if st.button("Login"):
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

st.sidebar.title(f"👋 Welcome, {st.session_state.username}")

# Logout Button
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

# Database Configuration
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

# -------------------------------------------
# 🔹 PAGE NAVIGATION
# -------------------------------------------
st.sidebar.title("📂 Navigation")
if st.sidebar.button("Live Insights Panel"):
    st.session_state.current_page = "Live Insights Panel"
    st.rerun()

if st.sidebar.button("Overview Panel"):
    st.session_state.current_page = "Overview Panel"
    st.rerun()

# -------------------------------------------
# 🔹 PAGE 1: Was Cached Distribution
# -------------------------------------------
if st.session_state.current_page == "Live Insights Panel":
    st.title("📊 Page 1: Was Cached Distribution")
    df = run_async_query("SELECT * FROM redset_main LIMIT 1000")

    cached_count = df["was_cached"].value_counts().reset_index()
    cached_count.columns = ["was_cached", "count"]
    fig = px.bar(cached_count, x="was_cached", y="count", title="Query Count by 'was_cached'")
    st.plotly_chart(fig)

# -------------------------------------------
# 🔹 PAGE 2: Top 20 Most Used Tables
# -------------------------------------------
elif st.session_state.current_page == "Overview Panel":
    st.title("📈 Page 2: Top 20 Most Used Tables")
    df = run_async_query("SELECT * FROM redset_main LIMIT 1000")

    fig, ax = plt.subplots()
    ax.bar(["Table1", "Table2"], [10, 20])  # Example data
    st.pyplot(fig)

    st.title("📊 Top K Tables")

    # Date range selector
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

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

    # -------------------------------------------
    # 🔹 Compile Time vs Number of Joins (With Log Transformation)
    # -------------------------------------------
    st.title("⏳ Compile Time vs Number of Joins (Log-Transformed)")

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
        ax.set_ylabel("Compile Duration (ms) (Log-Transformed)")
        ax.set_title("Compile Time vs Number of Joins (Log-Transformed)")
        ax.legend()
        
        st.pyplot(fig)

        st.write(f"📈 Best Fit Model: **{best_model}** with R² = **{best_r2:.3f}**")
        
    else:
        st.write("⚠️ No data available for Compile Time vs Number of Joins.")

# if st.session_state.current_page == "Live Insights Panel":
    st.title("📊 Page 2: Query Compilation vs Execution Time (Hourly Aggregation)")

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

#     import streamlit as st
# import plotly.express as px
# import pandas as pd

# if st.session_state.current_page == "Live Insights Panel":
    st.title("📊 Page 2: Cached vs Non-Cached Query Execution Time")

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
    st.title("📊 Page 2: Join vs Scan Efficiency")

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