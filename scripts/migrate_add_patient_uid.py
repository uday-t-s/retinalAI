#!/usr/bin/env python3
"""
Small one-off migration script to add `patient_uid` and `address` columns to the
existing SQLite `patient` table and populate `patient_uid` for existing rows.

Run from the project root:

    python3 scripts/migrate_add_patient_uid.py

This is safe to re-run (idempotent). It will:
 - add `address` column with default 'N/A' if missing
 - add `patient_uid` column if missing
 - populate `patient_uid` for rows where it's NULL/empty
 - create a unique index on `patient_uid`

Note: This modifies the SQLite file in-place. Back it up if you care about existing data.
"""

import sqlite3
import uuid
import os
import sys


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Try common locations for the SQLite file (project root and instance/ folder)
    candidates = [
        os.path.join(repo_root, 'site.db'),
        os.path.join(repo_root, 'instance', 'site.db'),
    ]

    db_path = None
    for c in candidates:
        if os.path.exists(c):
            db_path = c
            break

    if not db_path:
        print("ERROR: database file not found in expected locations:")
        for c in candidates:
            print("  -", c)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Check if table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='patient';")
    if not cur.fetchone():
        print("ERROR: table 'patient' does not exist in the database.")
        conn.close()
        sys.exit(1)

    # Get existing columns
    cur.execute("PRAGMA table_info(patient);")
    cols = [r[1] for r in cur.fetchall()]
    print("Existing patient columns:", cols)

    if 'address' not in cols:
        print("Adding 'address' column...")
        cur.execute("ALTER TABLE patient ADD COLUMN address TEXT DEFAULT 'N/A';")
        conn.commit()
    else:
        print("'address' column already present; skipping")

    if 'patient_uid' not in cols:
        print("Adding 'patient_uid' column...")
        # Add column without NOT NULL constraint (SQLite can't add NOT NULL with no default)
        cur.execute("ALTER TABLE patient ADD COLUMN patient_uid TEXT;")
        conn.commit()
    else:
        print("'patient_uid' column already present; skipping")

    # Populate missing patient_uid values
    cur.execute("SELECT id, patient_uid FROM patient WHERE patient_uid IS NULL OR patient_uid = '';")
    rows = cur.fetchall()
    if not rows:
        print("No existing rows need patient_uid population")
    else:
        print(f"Populating patient_uid for {len(rows)} rows...")
        for rid, existing in rows:
            new_uid = str(uuid.uuid4())
            cur.execute("UPDATE patient SET patient_uid = ? WHERE id = ?", (new_uid, rid))
        conn.commit()

    # Create unique index to enforce uniqueness going forward (will fail if duplicates)
    try:
        print("Creating unique index on patient_uid (if not exists)...")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_patient_patient_uid ON patient(patient_uid);")
        conn.commit()
    except Exception as e:
        print("Warning: could not create unique index on patient_uid:", e)

    print("Migration complete. You can now restart your Flask app.")
    conn.close()


if __name__ == '__main__':
    main()
