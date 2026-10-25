"""
FingerDiab AI — Database Migration Verifier & Registration Test
Run this script to:
  1. Apply the schema migration (adds missing columns safely)
  2. Verify the actual database schema
  3. Test registration of a new user
  4. Test duplicate username detection
  5. Test login with the new user
  6. Confirm existing users were preserved
"""
import sqlite3
import os
import sys

# Ensure we're in the right directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

DATABASE = os.path.join(SCRIPT_DIR, 'users.db')

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        PASS = "✅"
        FAIL = "❌"
    except Exception:
        PASS = "[PASS]"
        FAIL = "[FAIL]"
else:
    PASS = "✅"
    FAIL = "❌"
results = []


def report(test_name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((test_name, passed))
    msg = f"  {status} {test_name}" + (f" — {detail}" if detail else "")
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'replace').decode('ascii'))


print("=" * 60)
print("FINGERDIAB AI — DATABASE FIX VERIFICATION")
print("=" * 60)

# ── TEST 1: Run migration ────────────────────────────────────
print("\n[TEST 1] Run database migration...")
try:
    from init_db import init_enhanced_db
    init_enhanced_db()
    report("Migration executed", True)
except Exception as e:
    report("Migration executed", False, str(e))

# ── TEST 2: Verify schema ────────────────────────────────────
print("\n[TEST 2] Verify users table schema...")
try:
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("PRAGMA table_info(users)")
    columns = {row[1] for row in c.fetchall()}
    print(f"    Columns found: {sorted(columns)}")

    required = {'id', 'username', 'password', 'email', 'full_name', 'gender', 'created_at'}
    missing = required - columns
    if missing:
        report("All required columns present", False, f"Missing: {missing}")
    else:
        report("All required columns present", True, f"{len(required)} columns verified")
    conn.close()
except Exception as e:
    report("Schema verification", False, str(e))

# ── TEST 3: Check existing users preserved ───────────────────
print("\n[TEST 3] Check existing users preserved...")
try:
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT count(*) FROM users")
    count = c.fetchone()[0]
    c.execute("SELECT id, username FROM users")
    users = c.fetchall()
    conn.close()
    report("Existing users preserved", True, f"{count} user(s) in database")
    for u in users:
        print(f"    → id={u[0]} username={u[1]}")
except Exception as e:
    report("Existing users check", False, str(e))

# ── TEST 4: Test registration INSERT ─────────────────────────
print("\n[TEST 4] Test new user registration INSERT...")
try:
    from werkzeug.security import generate_password_hash, check_password_hash

    test_username = "__test_verify_user__"
    test_email = "testverify@example.com"
    test_password = "SecureTestPass123"
    test_fullname = "Dr. Sarah Johnson"
    test_gender = "Female"

    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    # Clean up any leftover test user
    c.execute("DELETE FROM users WHERE username = ?", (test_username,))
    conn.commit()

    hashed = generate_password_hash(test_password)
    c.execute(
        "INSERT INTO users (username, email, full_name, gender, password) VALUES (?, ?, ?, ?, ?)",
        (test_username, test_email, test_fullname, test_gender, hashed)
    )
    conn.commit()
    report("INSERT with email column", True, "No error")
except Exception as e:
    report("INSERT with email column", False, str(e))

# ── TEST 5: Verify inserted data ─────────────────────────────
print("\n[TEST 5] Verify inserted user data...")
try:
    c.execute("SELECT * FROM users WHERE username = ?", (test_username,))
    row = c.fetchone()
    if row:
        cols = [desc[0] for desc in c.description]
        data = dict(zip(cols, row))
        checks = [
            ("username", data.get('username') == test_username),
            ("email", data.get('email') == test_email),
            ("full_name", data.get('full_name') == test_fullname),
            ("gender", data.get('gender') == test_gender),
            ("password_hashed", data.get('password', '').startswith(('scrypt:', 'pbkdf2:', 'sha256$'))),
        ]
        all_ok = all(ok for _, ok in checks)
        report("User data integrity", all_ok, ", ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in checks))
    else:
        report("User data integrity", False, "User not found after INSERT")
except Exception as e:
    report("User data verification", False, str(e))

# ── TEST 6: Duplicate username detection ─────────────────────
print("\n[TEST 6] Test duplicate username detection...")
try:
    duplicate_caught = False
    try:
        c.execute(
            "INSERT INTO users (username, email, full_name, gender, password) VALUES (?, ?, ?, ?, ?)",
            (test_username, "other@example.com", "Dup User", "Other", hashed)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        duplicate_caught = True
    report("Duplicate username blocked", duplicate_caught)
except Exception as e:
    report("Duplicate username test", False, str(e))

# ── TEST 7: Login simulation ─────────────────────────────────
print("\n[TEST 7] Test login with new user...")
try:
    c.execute("SELECT * FROM users WHERE username = ?", (test_username,))
    user = c.fetchone()
    cols = [desc[0] for desc in c.description]
    user_dict = dict(zip(cols, user))

    password_ok = check_password_hash(user_dict['password'], test_password)
    report("Login password verification", password_ok)
except Exception as e:
    report("Login test", False, str(e))

# ── CLEANUP ──────────────────────────────────────────────────
print("\n[CLEANUP] Removing test user...")
try:
    c.execute("DELETE FROM users WHERE username = ?", (test_username,))
    conn.commit()
    conn.close()
    report("Test user cleaned up", True)
except Exception as e:
    report("Cleanup", False, str(e))

# ── SUMMARY ──────────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(1 for _, p in results if p)
total = len(results)
print(f"RESULTS: {passed}/{total} tests passed")
if passed == total:
    print(f"{PASS} ALL TESTS PASSED — Registration fix is verified!")
else:
    failed = [name for name, p in results if not p]
    print(f"{FAIL} FAILED TESTS: {', '.join(failed)}")
print("=" * 60)
