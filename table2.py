import psycopg2

# Database Configuration
DB_CONFIG = {
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432',
    'database': 'redset'
}

# SQL Query to Create or Refresh the Materialized View
SQL_QUERY = """
-- Drop the materialized view if it exists
DROP MATERIALIZED VIEW IF EXISTS public.hit_rate_per_day;

-- Create the materialized view
CREATE MATERIALIZED VIEW IF NOT EXISTS public.hit_rate_per_day
TABLESPACE pg_default
AS
WITH daily_stats AS (
    SELECT 
        date_trunc('day', redset_main.arrival_timestamp) AS day,
        redset_main.query_type,
        redset_main.user_id,
        COUNT(*) FILTER (WHERE redset_main.was_cached = 1) AS was_cached_count,
        COUNT(*) AS total_count
    FROM 
        redset_main
    GROUP BY 
        date_trunc('day', redset_main.arrival_timestamp),
        redset_main.query_type,
        redset_main.user_id
),
daily_totals AS (
    SELECT 
        day,
        SUM(total_count) AS total_queries
    FROM 
        daily_stats
    GROUP BY 
        day
)
SELECT 
    ds.day,
    ds.query_type,
    ds.user_id,
    ds.was_cached_count,
    dt.total_queries,
    (ds.was_cached_count::double precision / NULLIF(dt.total_queries, 0)::double precision) * 100 AS hit_rate_per_day
FROM 
    daily_stats ds
JOIN 
    daily_totals dt
ON 
    ds.day = dt.day
WITH DATA;

-- Create a unique index on the materialized view
CREATE UNIQUE INDEX idx_hit_rate_per_day ON public.hit_rate_per_day (day, query_type, user_id);

-- Refresh the materialized view concurrently
REFRESH MATERIALIZED VIEW CONCURRENTLY public.hit_rate_per_day;
"""

# Function to execute SQL query
def execute_query():
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Execute the query
        cursor.execute(SQL_QUERY)

        # Commit changes
        conn.commit()
        print("✅ Materialized view 'hit_rate_per_day' created/refreshed successfully.")

    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
    
    finally:
        # Close cursor and connection
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        print("🔌 Database connection closed.")

# Run the function
execute_query()