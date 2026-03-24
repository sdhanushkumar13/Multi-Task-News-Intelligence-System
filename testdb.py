import psycopg2
import os
from datetime import datetime

DB_HOST = "nlp-db.cx6w4cqw67ox.ap-south-1.rds.amazonaws.com"
DB_NAME = "nlp_logs"
DB_USER = "postgres"
DB_PASSWORD = "Dk23awsrds"

print("Connecting...")

try:
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=5432,
        connect_timeout=10
    )
    print("Connected!")

    cur = conn.cursor()
    print("Cursor created!")

    cur.execute("""
        INSERT INTO inference_logs (
            user_id, task_type, model_family, model_name, input_length, output, error_flag
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, ("test_user", "Classification", "ML", "logreg_tfidf", 100, "test_output", False))

    print("Query executed!")

    conn.commit()
    print("Committed! ✅ Data inserted successfully")

    cur.close()
    conn.close()
    print("Connection closed.")

except Exception as e:
    print(f"ERROR TYPE: {type(e)._name_}")
    print(f"ERROR DETAIL: {str(e)}")
