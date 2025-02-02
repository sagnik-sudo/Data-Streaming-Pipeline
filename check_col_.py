import duckdb
import pandas as pd

# Path to your Parquet file
parquet_file = "full.parquet"

# Query to extract day-wise record count for March
query = f"""
SELECT 
    strftime('%Y-%m-%d', arrival_timestamp) AS day, 
    COUNT(*) AS record_count
FROM '{parquet_file}'
WHERE strftime('%Y-%m', arrival_timestamp) = '2024-03'  -- Change year if needed
GROUP BY day
ORDER BY day;
"""

# Execute query and store result
df_march = duckdb.query(query).to_df()

# Display result
print(df_march)