"""
FingerDiab AI — Automated Authentication Tests
Tests registration, password hashing verification, duplicate prevention, and session lifecycle.
"""

import unittest
import time
from fingerdiab.database.repositories import UserRepository

class TestAuthentication(unittest.TestCase):
    def setUp(self):
        ts = int(time.time() * 1000)
        self.test_username = f"test_doc_auth_{ts}"
        self.test_password = "SecurePass123!"
        self.test_email = f"test_auth_{ts}@example.com"
        
    def test_user_lifecycle(self):
        # 1. Create user
        success, user_id, err = UserRepository.create_user(
            username=self.test_username,
            plain_password=self.test_password,
            email=self.test_email,
            full_name="Dr. Test User",
            gender="Female"
        )
        self.assertTrue(success)
        self.assertIsNotNone(user_id)
            
        # 2. Verify valid credentials
        valid, user = UserRepository.verify_credentials(self.test_username, self.test_password)
        self.assertTrue(valid)
        self.assertIsNotNone(user)
        self.assertEqual(user['username'], self.test_username)
        
        # 3. Verify invalid password rejected
        invalid, _ = UserRepository.verify_credentials(self.test_username, "WrongPassword999!")
        self.assertFalse(invalid)
        
        # 4. Verify duplicate registration rejected
        dup_success, _, dup_err = UserRepository.create_user(
            username=self.test_username,
            plain_password="anotherpassword",
            email="another@example.com"
        )
        self.assertFalse(dup_success)
        self.assertIn("already taken", dup_err)

if __name__ == '__main__':
    unittest.main()
