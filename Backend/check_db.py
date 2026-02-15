import os
import psycopg2
from urllib.parse import urlparse

# Force load .env for local testing
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def check_db():
    print(f"Checking Database Connection...")
    
    if not DATABASE_URL:
        print("❌ DATABASE_URL is not set!")
        return

    # Mask specific parts of URL for safety
    masked_url = DATABASE_URL
    if "@" in DATABASE_URL:
        masked_url = DATABASE_URL.split("@")[1]
    print(f"URL Host: {masked_url}")

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Check current user and DB
        cursor.execute("SELECT current_user, current_database();")
        user, db = cursor.fetchone()
        print(f"✅ Connected as user '{user}' to database '{db}'")

        # Check tables in public schema
        cursor.execute("""
            SELECT tablename 
            FROM pg_catalog.pg_tables 
            WHERE schemaname = 'public';
        """)
        tables = [row[0] for row in cursor.fetchall()]
        print(f"📋 Found tables: {tables}")

        if "jobs" in tables:
            print("✅ 'jobs' table exists!")
            # Check schema details
            cursor.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'jobs';")
            columns = cursor.fetchall()
            print("   Columns:")
            for col in columns:
                print(f"   - {col[0]} ({col[1]})")
        else:
            print("❌ 'jobs' table MISSING!")
            print("   Attemping to create it now...")
            try:
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    filename TEXT,
                    message TEXT,
                    progress INTEGER DEFAULT 0,
                    result_path TEXT,
                    error TEXT,
                    report_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
                conn.commit()
                print("✅ 'jobs' table created successfully!")
            except Exception as e:
                print(f"❌ Failed to create table: {e}")

        conn.close()

    except Exception as e:
        print(f"❌ Connection Failed: {e}")

if __name__ == "__main__":
    check_db()
