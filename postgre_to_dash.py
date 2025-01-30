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
    st.session_state.current_page = "Page 1"  # Default to Page 1

# -------------------------------------------
# 🔹 LOGIN & REGISTRATION PAGE
# -------------------------------------------
if not st.session_state.logged_in:
    st.title("🔐 Login / Register Page")
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
    st.title("🔴 Admin Dashboard")
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
if st.sidebar.button("Go to Page 1"):
    st.session_state.current_page = "Page 1"
    st.rerun()

if st.sidebar.button("Go to Page 2"):
    st.session_state.current_page = "Page 2"
    st.rerun()

# -------------------------------------------
# 🔹 PAGE 1: Was Cached Distribution
# -------------------------------------------
if st.session_state.current_page == "Page 1":
    st.title("📊 Page 1: Was Cached Distribution")
    df = run_async_query("SELECT * FROM redset_raw LIMIT 1000")

    cached_count = df["was_cached"].value_counts().reset_index()
    cached_count.columns = ["was_cached", "count"]
    fig = px.bar(cached_count, x="was_cached", y="count", title="Query Count by 'was_cached'")
    st.plotly_chart(fig)

# -------------------------------------------
# 🔹 PAGE 2: Top 20 Most Used Tables
# -------------------------------------------
elif st.session_state.current_page == "Page 2":
    st.title("📈 Page 2: Top 20 Most Used Tables")
    df = run_async_query("SELECT * FROM redset_raw LIMIT 1000")

    fig, ax = plt.subplots()
    ax.bar(["Table1", "Table2"], [10, 20])  # Example data
    st.pyplot(fig)