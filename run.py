"""
FingerDiab AI — Application Entry Point
Starts the Flask server with pre-loaded models and database initialization.
"""

import os
import sys
import socket

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app
from fingerdiab.config import Config

def find_free_port(preferred_port=5000):
    for port in [preferred_port, 5005, 5050, 8000, 8080]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return port
            except OSError:
                continue
    return preferred_port

if __name__ == '__main__':
    default_port = int(os.environ.get('PORT', 5005))
    port = find_free_port(default_port)
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    print("=" * 65)
    print("  FINGERDIAB AI — BIOMETRIC DIABETES RISK SCREENING SYSTEM")
    print(f"  Server URL: http://127.0.0.1:{port}")
    print("=" * 65)
    app.run(host='0.0.0.0', port=port, debug=debug)
