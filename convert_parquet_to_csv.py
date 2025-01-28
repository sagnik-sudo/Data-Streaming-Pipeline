import duckdb

# Path to your Parquet file
parquet_file = 'sample_0.001.parquet'

# Path for the output CSV file
csv_file = 'output.csv'

# Run the conversion
duckdb.query(f"COPY (SELECT * FROM '{parquet_file}') TO '{csv_file}' (FORMAT CSV, HEADER TRUE)")

print(f"Converted {parquet_file} to {csv_file}")