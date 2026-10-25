"""
FingerDiab AI — Application Configuration
Loads environment variables securely with production defaults.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    BASE_DIR = BASE_DIR
    SECRET_KEY = os.environ.get('SECRET_KEY', 'fingerdiab-research-secret-default-key-2026')
    DATABASE_PATH = os.environ.get('DATABASE_PATH', os.path.join(BASE_DIR, 'users.db'))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MODEL_DIR = os.path.join(BASE_DIR, 'model')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024 # 16 MB
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    MODEL_VERSION = 'fingerdiab-rf-v1.0.0'
