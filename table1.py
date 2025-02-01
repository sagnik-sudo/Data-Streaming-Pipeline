import asyncio
import asyncpg

DB_CONFIG = {
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432',
    'database': 'redset'
}

CREATE_VIEW_QUERY = """
CREATE MATERIALIZED VIEW top_k_tables_per_day AS
WITH table_usage AS (
    SELECT
        arrival_timestamp::TIMESTAMP,  -- Explicitly cast to TIMESTAMP
        date_trunc('day', arrival_timestamp::TIMESTAMP) AS day,  -- Casted
        unnest(string_to_array(read_table_ids, ',')) AS table_id
    FROM
        public.redset_main
    UNION ALL
    SELECT
        arrival_timestamp::TIMESTAMP,  -- Explicitly cast to TIMESTAMP
        date_trunc('day', arrival_timestamp::TIMESTAMP) AS day,  -- Casted
        unnest(string_to_array(write_table_ids, ',')) AS table_id
    FROM
        public.redset_main
),
table_count AS (
    SELECT
        day,
        table_id,
        COUNT(*) AS count
    FROM
        table_usage
    GROUP BY
        day, table_id
),
total_count AS (
    SELECT
        day,
        SUM(count) AS total
    FROM
        table_count
    GROUP BY
        day
),
overall_total AS (
    SELECT
        SUM(total) AS overall_total
    FROM
        total_count
),
table_percentage AS (
    SELECT
        day,
        table_id,
        count,
        (count::float / (SELECT total FROM total_count WHERE total_count.day = table_count.day)) * 100 AS percentage,
        (count::float / (SELECT overall_total FROM overall_total)) * 100 AS overall_percentage
    FROM
        table_count
)
SELECT
    arrival_timestamp::TIMESTAMP,  -- Explicitly cast to TIMESTAMP
    day,
    table_id AS Table_ID,
    count AS Count,
    percentage,
    overall_percentage
FROM
    table_usage
JOIN
    table_percentage USING (day, table_id)
ORDER BY
    day, count DESC;
"""

async def create_materialized_view():
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        await conn.execute(CREATE_VIEW_QUERY)
        await conn.close()
        print("✅ Materialized view created successfully!")
    except Exception as e:
        print(f"❌ Error creating materialized view: {e}")

# Run the async function
asyncio.run(create_materialized_view())