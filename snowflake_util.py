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


def load_posts_to_staging(posts_df, con):
    print("\nLoading data into Snowflake staging table...")
    with con.cursor() as cur:
        cur.execute("USE SCHEMA RAW")
        cur.execute("TRUNCATE TABLE posts_stage")

    success, _, nrows, _ = write_pandas(
        con, posts_df, "POSTS_STAGE", quote_identifiers=False
    )
    if success:
        print(f"Successfully loaded {nrows} rows into Snowflake")
    else:
        print("Failed to load data into Snowflake")


def merge_to_posts(con):
    print("\nMerging data from staging to main table...")
    with con.cursor() as cur:
        cur.execute("USE SCHEMA RAW")
        cur.execute(
            """
                MERGE INTO POSTS p
            USING POSTS_STAGE s
            ON p.post_id = s.post_id
            WHEN MATCHED THEN UPDATE SET
                p.topic = s.subreddit,
                p.keywords = s.keywords,
                p.title = s.title,
                p.selftext = s.selftext,
                p.text = TRIM(CONCAT(s.title, ' ', s.selftext)),
                p.created_utc = TRY_TO_TIMESTAMP_NTZ(s.created_utc),
                p.permalink = s.permalink,
                p.updated_at = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN INSERT (
                post_id,
                topic,
                keywords,
                title, 
                selftext, 
                text, 
                created_utc, 
                permalink, 
                updated_at
            )
            VALUES (
                s.post_id, 
                s.subreddit, 
                s.keywords, 
                s.title, 
                s.selftext, 
                TRIM(CONCAT(s.title, ' ', s.selftext)), 
                TRY_TO_TIMESTAMP_NTZ(s.created_utc), 
                s.permalink, 
                CURRENT_TIMESTAMP()
            )
        """
        )
        merge_qid = cur.sfqid
        print("Merge completed")
        cur.execute(
            f"""
                    SELECT
                        "number of rows inserted" as rows_inserted,
                        "number of rows updated" as rows_updated,
                    FROM TABLE(RESULT_SCAN('{merge_qid}'))
        """
        )
        result = cur.fetchone()
        print(f"Rows inserted: {result[0]}, Rows updated: {result[1]}")
        return result[0]


def get_posts_embeddings(con):
    print("\nFetching posts with embeddings from Snowflake...")
    with con.cursor() as cur:
        cur.execute("USE SCHEMA VECTOR")
        cur.execute("SELECT * FROM POSTS_EMBEDDINGS")
        posts_df = cur.fetch_pandas_all()
    print(f"Fetched {len(posts_df)} posts with embeddings from Snowflake")
    return posts_df


def set_model_status(con, status):
    with con.cursor() as cur:
        cur.execute("USE SCHEMA STATUS")
        cur.execute(
            f"""
                    UPDATE MODEL_STATUS 
                    SET UPDATE_STATUS = {status},
                        LAST_UPDATED = CURRENT_TIMESTAMP()
                    WHERE TRUE"""
        )


def get_model_status(con):
    with con.cursor() as cur:
        cur.execute("USE SCHEMA STATUS")
        cur.execute("SELECT UPDATE_STATUS FROM MODEL_STATUS")
        result = cur.fetchone()
        return result[0]


def embed_text(con, text):
    with con.cursor() as cur:
        cur.execute(
            f"""SELECT 
                    SNOWFLAKE.CORTEX.EMBED_TEXT_1024(
                        'snowflake-arctic-embed-l-v2.0',
                        '{text}'
                    )"""
        )
        result = cur.fetchone()
        return result[0]
