import duckdb

# Define the path to the final sorted file
final_sorted_file = "/Volumes/SD SSD/tmp_and_parquet/sorted.parquet"

# Define external storage for DuckDB temp files
duckdb_temp_dir = "/Volumes/SD SSD/tmp_and_parquet/duckdb_tmp"

# Connect to DuckDB with external storage for temp files
con = duckdb.connect()
con.execute(f"SET temp_directory = '{duckdb_temp_dir}'")  # Use external SSD for temp storage
con.execute("SET memory_limit = '5.5GB'")  # Limit memory usage
con.execute("PRAGMA enable_progress_bar")  # Show progress

print("🚀 Checking if sorting is correct... (Efficiently)")

# Check the first and last timestamps (basic validation)
first_last = con.execute(f"""
    SELECT 
        MIN(arrival_timestamp) AS first_timestamp,
        MAX(arrival_timestamp) AS last_timestamp
    FROM read_parquet('{final_sorted_file}')
""").fetchone()

print(f"🕒 First timestamp: {first_last[0]}")
print(f"🕒 Last timestamp: {first_last[1]}")

# Efficiently check for out-of-order timestamps
print("🔄 Scanning the file to detect sorting issues (this may take some time)...")

out_of_order = con.execute(f"""
    SELECT COUNT(*) 
    FROM (
        SELECT arrival_timestamp, 
               LAG(arrival_timestamp) OVER (ORDER BY arrival_timestamp) AS prev_timestamp
        FROM read_parquet('{final_sorted_file}')
    ) 
    WHERE arrival_timestamp < prev_timestamp
""").fetchone()[0]

if out_of_order == 0:
    print("✅ Sorting is correct! The file is in ascending order.")
else:
    print(f"❌ Sorting is incorrect! Found {out_of_order} out-of-order timestamps.")

# Close DuckDB connection
con.close()