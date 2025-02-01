import asyncio
import asyncpg
import duckdb
import pandas as pd

# PostgreSQL Database Configuration
DB_CONFIG = {
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432',
    'database': 'redset'
}

PARQUET_FILE = "full.parquet"
TABLE_NAME = "redset_main"
MAX_ROWS = 100000  # Limit to 1000 rows

async def create_table(conn, df):
    """Creates a table dynamically based on DataFrame schema."""
    column_definitions = []
    for column, dtype in df.dtypes.items():
        dtype_str = str(dtype)
        if dtype_str.startswith("int64"):
            pg_type = "BIGINT"
        elif dtype_str.startswith("int32"):
            pg_type = "INTEGER"
        elif dtype_str.startswith("float64"):
            pg_type = "DOUBLE PRECISION"
        elif dtype_str.startswith("float32"):
            pg_type = "REAL"
        elif dtype_str.startswith("bool"):
            pg_type = "BOOLEAN"
        elif dtype_str.startswith("datetime64"):
            pg_type = "TIMESTAMP"
        elif dtype_str.startswith("timedelta64"):
            pg_type = "INTERVAL"
        elif dtype_str.startswith("category") or dtype_str.startswith("string"):
            pg_type = "TEXT"
        elif dtype_str.startswith("uint"):
            pg_type = "BIGINT"
        else:
            pg_type = "TEXT"

        column_definitions.append(f'"{column}" {pg_type}')

    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        {', '.join(column_definitions)}
    );
    """
    await conn.execute(create_table_query)

async def insert_data():
    """Loads 1000 rows from Parquet using DuckDB and inserts into PostgreSQL."""
    
    # Query only the first 1000 rows using DuckDB (Efficient)
    query = f"SELECT * FROM read_parquet('{PARQUET_FILE}') LIMIT {MAX_ROWS}"
    df = duckdb.query(query).to_df()

    if df.empty:
        print("⚠️ No data found in Parquet file!")
        return

    # Establish asyncpg connection
    conn = await asyncpg.connect(**DB_CONFIG)

    # Ensure table exists
    await create_table(conn, df)

    # Convert DataFrame rows to tuples for bulk insert
    records = list(df.itertuples(index=False, name=None))

    # Generate SQL insert statement dynamically
    columns = ', '.join([f'"{col}"' for col in df.columns])
    values_placeholder = ', '.join([f"${i+1}" for i in range(len(df.columns))])
    insert_query = f"INSERT INTO {TABLE_NAME} ({columns}) VALUES ({values_placeholder})"

    # Bulk insert all rows
    await conn.executemany(insert_query, records)

    # Close connection
    await conn.close()
    print(f"✅ Successfully inserted {len(records)} rows into {TABLE_NAME}!")

# Run the async function
asyncio.run(insert_data())