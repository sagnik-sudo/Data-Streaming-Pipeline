import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Custom CSS for fonts
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    /* Apply Roboto font to the entire app */
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the Dashboard
st.title("Dashboard on Streamlit")

# Generating sample data for the three graphs
# Data for Graph 1: Bar Graph (only first 9 rows)
table_data = pd.DataFrame({
    "Table ID": [f"Table_{i}" for i in range(1, 11)],
    "Count": np.sort(np.random.randint(10, 100, 10))[::-1]  # Sorted in decreasing order
}).iloc[:9]  # Slice to keep only the first 9 rows

# Data for Graph 2: Line Graph with 100 Data Points and Custom Pattern
hit_rate = []
for i in range(1, 101):
    if i == 1:
        hit_rate.append(100)  # Start at 100%
    elif i <= 20:
        hit_rate.append(max(10, 100 - i * 5))  # Sharp drop
    elif 20 < i <= 40:
        hit_rate.append(min(15, 10 + i % 5))  # Gradual increase to around 10%
    elif 40 < i <= 70:
        hit_rate.append(np.random.uniform(5, 15))  # Fluctuation between 5% and 15%
    else:
        hit_rate.append(10)  # Stabilize around 10%

query_data = pd.DataFrame({
    "Query Count": np.arange(1, 101),  # 1 to 100
    "Hit Rate %": hit_rate
})

# Data for Graph 3: Dotted Graph (only first 7 rows)
joins_data = pd.DataFrame({
    "Number of Joins": np.arange(1, 11),  # 1 to 10
    "Compile Time (ms)": np.sort(np.random.uniform(10, 100, 10))  # Sorted increasing
}).iloc[:7]  # Slice to keep only the first 7 rows

# Layout: Side-by-Side with st.columns
col1, col2 = st.columns(2)

# First Graph: Bar Graph in col1
with col1:
    st.subheader("Top K Tables")
    st.bar_chart(table_data.set_index("Table ID")["Count"])

# Second Graph: Line Graph in col2
with col2:
    st.subheader("Hit Rate of Result Cache")
    st.line_chart(query_data.set_index("Query Count")["Hit Rate %"])

# Third Graph: Dotted Graph below
st.subheader("Relation b/w Join Count and Compile Time")
fig, ax = plt.subplots()
ax.plot(
    joins_data["Number of Joins"],
    joins_data["Compile Time (ms)"],
    marker='o', linestyle='-', color='blue'
)
ax.set_title("Relation b/w Join Count and Compile Time")
ax.set_xlabel("Number of Joins")
ax.set_ylabel("Compile Time (ms)")

st.pyplot(fig)