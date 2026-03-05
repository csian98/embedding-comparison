import os
from dotenv import load_dotenv
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

load_dotenv()


def get_connection():
    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        role=os.getenv("SNOWFLAKE_ROLE"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
    )


def get_posts(con):
    print("\nFetching posts from Snowflake...")
    with con.cursor() as cur:
        cur.execute("USE SCHEMA RAW")
        cur.execute("SELECT * FROM POSTS")
        posts_df = cur.fetch_pandas_all()
    print(f"Fetched {len(posts_df)} posts from Snowflake")
    return posts_df


def get_posts_embeddings(con):
    print("\nFetching posts with embeddings from Snowflake...")
    with con.cursor() as cur:
        cur.execute("USE SCHEMA VECTOR")
        cur.execute("SELECT * FROM POSTS_EMBEDDINGS")
        posts_df = cur.fetch_pandas_all()
    print(f"Fetched {len(posts_df)} posts with embeddings from Snowflake")
    return posts_df



