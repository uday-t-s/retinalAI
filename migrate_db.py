import sqlite3
import os

DB_PATH = 'instance/site.db'

def migrate():
    if not os.path.exists(DB_PATH):
        print(f"Database {DB_PATH} not found!")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # 1. Add is_blocked to user table
        print("Checking 'user' table for 'is_blocked' column...")
        cursor.execute("PRAGMA table_info(user)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'is_blocked' not in columns:
            print("Adding 'is_blocked' column to 'user' table...")
            cursor.execute("ALTER TABLE user ADD COLUMN is_blocked BOOLEAN DEFAULT 0 NOT NULL")
            print("Done.")
        else:
            print("'is_blocked' column already exists.")

        # 2. Add user_id to patient table
        print("Checking 'patient' table for 'user_id' column...")
        cursor.execute("PRAGMA table_info(patient)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'user_id' not in columns:
            print("Adding 'user_id' column to 'patient' table...")
            cursor.execute("ALTER TABLE patient ADD COLUMN user_id INTEGER REFERENCES user(id)")
            print("Done.")
        else:
            print("'user_id' column already exists.")

        conn.commit()
        print("Migration completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == '__main__':
    migrate()
