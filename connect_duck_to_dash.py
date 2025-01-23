from sqlalchemy import create_engine

# Path to the DuckDB database file
db_path = "/Users/sagnikdas/GitHub/Data-Streaming-Pipeline/redset.duckdb"

# SQLAlchemy connection string
engine = create_engine(f"duckdb:///{db_path}")

# Example: Execute a query
with engine.connect() as connection:
    result = connection.execute("SELECT 1")
    for row in result:
        print(row)