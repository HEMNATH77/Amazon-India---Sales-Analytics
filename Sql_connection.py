import pandas as pd
import pymysql
from sqlalchemy import create_engine


#  Load CSV file ---
df = pd.read_csv("Amazon_final_Dataset.csv")




# MySQL connection details
username = 'root'
password = 'root'
host = 'localhost'
port = 3306
database = 'amazon_sales_analytics'
table_name = 'amazon_orders'

# Create engine using pymysql as the connector
engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}")

# Write DataFrame to MySQL
df.to_sql(
    name=table_name,
    con=engine,
    if_exists='append',   # use 'replace' to drop and recreate table
    index=False,
    chunksize=1000,
    method='multi'        # sends many rows per insert for speed
)

print("✅ DataFrame successfully inserted into MySQL table!")
