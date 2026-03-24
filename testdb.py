import streamlit as st
import psycopg2
import os

# --------------------------------
# RDS SETUP - HARDCODED FOR TEST
# --------------------------------
DB_HOST = "nlp-db.cx6w4cqw67ox.ap-south-1.rds.amazonaws.com"
DB_NAME = "nlp_logs"
DB_USER = "postgres"
DB_PASSWORD = "xxxxxxxxxx" # Replace with your real password string
DB_PORT = 5432

st.set_page_config(page_title="DB Debugger", layout="centered")
st.title("🐘 RDS Connection Diagnostic")

st.info(f"Attempting to connect to: **{DB_HOST}**")

# Use a standard function without @st.cache_resource for testing
def test_connection():
    conn = None
    try:
        st.write("1. Initializing connection...")
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            connect_timeout=10  # Forces a failure after 10s instead of infinite hang
        )
        
        st.write("2. Connection established. Creating cursor...")
        cur = conn.cursor()
        
        st.write("3. Executing test query...")
        cur.execute("SELECT current_database(), now();")
        result = cur.fetchone()
        
        st.success(f"✅ Success! Connected to: {result[0]}")
        st.write(f"Server Time: {result[1]}")
        
        cur.close()
    except psycopg2.OperationalError as e:
        st.error("❌ Operational Error (Network/Auth)")
        st.code(str(e))
    except Exception as e:
        st.error("❌ Unexpected Error")
        st.code(str(e))
    finally:
        if conn:
            conn.close()
            st.write("4. Connection closed safely.")

# UI Button to trigger the logic
if st.button("Run DB Handshake"):
    with st.spinner("Connecting..."):
        test_connection()

st.markdown("---")
st.subheader("Troubleshooting Steps if it hangs:")
st.markdown("""
1. **Security Group:** Is your EC2's **Private IP** or **Security Group ID** allowed in the RDS Inbound rules?
2. **VPC:** Are the EC2 and RDS in the same VPC?
3. **Public Access:** If the RDS is set to `Publicly Accessible: Yes`, try adding `0.0.0.0/0` to the RDS inbound rules *temporarily* to see if the timeout disappears.
""")
