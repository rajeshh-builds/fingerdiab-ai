"""
FingerDiab AI — Security & Cryptography Utilities
Provides secure password hashing, path traversal defense, and file extension checks.
"""

import os
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
MAX_FILE_SIZE_BYTES = 16 * 1024 * 1024 # 16 MB

def hash_password(plain_text):
    """Generates strong PBKDF2:SHA256 salted hash."""
    return generate_password_hash(plain_text, method='pbkdf2:sha256')

def verify_password(stored_hash, plain_text):
    """Verifies plain text password against stored hash."""
    if not stored_hash or not plain_text:
        return False
    return check_password_hash(stored_hash, plain_text)

def is_allowed_file(filename):
    """Checks if uploaded filename has an approved image extension."""
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def sanitize_and_validate_upload_path(filename, upload_dir):
    """
    Prevents path traversal attacks by ensuring sanitized filename resides strictly
    within designated upload directory.
    """
    safe_name = secure_filename(filename)
    if not safe_name:
        return None, "Invalid file name."
        
    full_path = os.path.abspath(os.path.join(upload_dir, safe_name))
    upload_root = os.path.abspath(upload_dir)
    
    # Path traversal check
    if not full_path.startswith(upload_root):
        return None, "Unauthorized path traversal attempt detected."
        
    return full_path, None
