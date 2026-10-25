"""
FingerDiab AI — Database Infrastructure & Connection Management
Provides safe, parameterized SQLite connection lifecycle and schema migration.
"""

import sqlite3
import os
from fingerdiab.config import Config

def get_db_connection():
    """Returns SQLite connection with row factory enabled."""
    conn = sqlite3.connect(Config.DATABASE_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes and migrates database tables with safe column checks."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Users Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT,
            full_name TEXT,
            gender TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Safe column migrations for users
    cursor.execute("PRAGMA table_info(users)")
    existing_user_cols = [row['name'] for row in cursor.fetchall()]
    if 'email' not in existing_user_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN email TEXT")
    if 'full_name' not in existing_user_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN full_name TEXT")
    if 'gender' not in existing_user_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN gender TEXT")
        
    # 2. Predictions Table
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
            risk_score REAL NOT NULL,
            model_used TEXT NOT NULL,
            confidence REAL,
            model_version TEXT DEFAULT 'fingerdiab-rf-v1.0.0',
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Safe column migrations for predictions
    cursor.execute("PRAGMA table_info(predictions)")
    existing_pred_cols = [row['name'] for row in cursor.fetchall()]
    if 'model_version' not in existing_pred_cols:
        cursor.execute("ALTER TABLE predictions ADD COLUMN model_version TEXT DEFAULT 'fingerdiab-rf-v1.0.0'")
    if 'confidence' not in existing_pred_cols:
        cursor.execute("ALTER TABLE predictions ADD COLUMN confidence REAL")
    if 'risk_score' not in existing_pred_cols:
        cursor.execute("ALTER TABLE predictions ADD COLUMN risk_score REAL")
        
    # 3. Consultations / Feedback Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS consultations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT,
            message TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 4. Audit Log Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT NOT NULL,
            ip_address TEXT,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
