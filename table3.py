import asyncio
import asyncpg

DB_CONFIG = {
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432',
    'database': 'redset'
}

CREATE_MATERIALIZED_VIEW_QUERY = """
DROP MATERIALIZED VIEW compile_time_vs_num_joins CASCADE;
CREATE MATERIALIZED VIEW public.compile_time_vs_num_joins AS
SELECT 
    num_joins AS x,
    AVG(compile_duration_ms) AS y
FROM public.redset_main
WHERE query_type = 'select' 
AND num_joins IS NOT NULL
GROUP BY num_joins
ORDER BY num_joins;
"""

async def create_materialized_view():
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        await conn.execute(CREATE_MATERIALIZED_VIEW_QUERY)
        await conn.close()
        print("✅ Materialized view created successfully!")
    except Exception as e:
        print(f"❌ Error creating materialized view: {e}")

# Run the async function
if __name__ == "__main__":
    asyncio.run(create_materialized_view())