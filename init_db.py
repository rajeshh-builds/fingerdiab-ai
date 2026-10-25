import os
import sqlite3

DATABASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.db')

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def _get_existing_columns(cursor, table_name):
    """Return a set of column names that already exist in a table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cursor.fetchall()}


def _migrate_users_table(cursor):
    """
    Safely migrate the users table from the old 3-column schema
    (id, username, password) to the full schema by adding any
    missing columns via ALTER TABLE.

    SQLite supports ALTER TABLE ... ADD COLUMN safely.
    Each ALTER is guarded by checking whether the column already exists,
    making this fully idempotent.
    """
    existing = _get_existing_columns(cursor, 'users')

    migrations = [
        # (column_name, column_definition)
        ('email',      'TEXT DEFAULT NULL'),
        ('full_name',  'TEXT DEFAULT NULL'),
        ('gender',     'TEXT DEFAULT NULL'),
        ('created_at', 'TIMESTAMP DEFAULT NULL'),
    ]

    for col_name, col_def in migrations:
        if col_name not in existing:
            cursor.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_def}")
            if col_name == 'created_at':
                cursor.execute("UPDATE users SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")
            print(f"  [MIGRATION] Added column 'users.{col_name}'")

    # Ensure future inserts without explicit created_at have timestamp populated
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS trg_users_created_at
        AFTER INSERT ON users
        FOR EACH ROW
        WHEN NEW.created_at IS NULL
        BEGIN
            UPDATE users SET created_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END;
    ''')


def init_enhanced_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # ── Users table ──────────────────────────────────────────────
    # CREATE TABLE IF NOT EXISTS creates the FULL schema on a fresh DB.
    # On an existing DB with the old schema, it is a no-op, so we
    # follow up with _migrate_users_table() to add missing columns.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password TEXT NOT NULL,
            full_name TEXT,
            gender TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    _migrate_users_table(cursor)

    # ── Predictions History table ────────────────────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            patient_name TEXT,
            fingerprint_type TEXT NOT NULL,
            ridge_count INTEGER NOT NULL,
            age INTEGER NOT NULL,
            bmi REAL NOT NULL,
            family_history TEXT NOT NULL,
            diabetes_risk TEXT NOT NULL,
            risk_score REAL DEFAULT 0.0,
            model_used TEXT DEFAULT 'Random Forest',
            confidence REAL DEFAULT 0.0,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # ── Feedback / Consultation queries table ────────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS consultations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT,
            message TEXT NOT NULL,
            status TEXT DEFAULT 'Pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    print("Enhanced database schema initialized successfully.")


if __name__ == '__main__':
    init_enhanced_db()
