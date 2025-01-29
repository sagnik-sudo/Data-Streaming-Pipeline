from pyspark.sql import SparkSession
from sqlalchemy import create_engine, text

# Database connection details
db_user = 'postgres'
db_password = 'postgres'
db_host = 'localhost'
db_port = '5432'
db_name = 'redset'
table_name = 'redset_raw'

# Create a Spark session
spark = SparkSession.builder \
    .appName("LoadParquetToPostgreSQL") \
    .config("spark.jars", "/Users/sagnikdas/spark_jars/postgresql-42.7.5.jar") \
    .getOrCreate()

# Path to the Parquet file
parquet_file_path = './full_sl.parquet'

# Read the Parquet file into a Spark DataFrame
df = spark.read.parquet(parquet_file_path)

# Create a database connection using SQLAlchemy
engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

# Generate the CREATE TABLE statement based on the DataFrame schema
def generate_create_table_query(df, table_name):
    dtype_mapping = {
        'LongType': 'BIGINT',
        'DoubleType': 'DOUBLE PRECISION',
        'StringType': 'TEXT',
        'TimestampType': 'TIMESTAMP',
        'BooleanType': 'BOOLEAN'
    }
    columns = []
    for field in df.schema.fields:
        sql_dtype = dtype_mapping.get(field.dataType.simpleString(), 'TEXT')
        columns.append(f"{field.name} {sql_dtype}")
    columns_sql = ",\n    ".join(columns)
    return f"CREATE TABLE IF NOT EXISTS {table_name} (\n    {columns_sql}\n);"

create_table_query = generate_create_table_query(df, table_name)

# Execute the CREATE TABLE statement
with engine.connect() as conn:
    conn.execute(text(create_table_query))

# Write the DataFrame to PostgreSQL in chunks
df.write \
    .format("jdbc") \
    .option("url", f"jdbc:postgresql://{db_host}:{db_port}/{db_name}") \
    .option("dbtable", table_name) \
    .option("user", db_user) \
    .option("password", db_password) \
    .option("driver", "org.postgresql.Driver") \
    .mode("append") \
    .save()

print('Finished loading the Parquet file into PostgreSQL')