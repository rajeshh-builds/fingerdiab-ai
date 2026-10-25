"""
FingerDiab AI — Automated End-to-End Prediction Tests
Tests full web flow integration from form submission to numerical dossier generation and lifestyle recommendation rendering.
"""

import unittest
from app import app
from fingerdiab.database.repositories import UserRepository

class TestPredictionIntegration(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.username = "doc_integration_test"
        self.password = "DoctorPass2026!"
        
        UserRepository.create_user(self.username, self.password, "doc@test.com", "Dr. Tester", "Male")
        
        # Login
        with self.app.session_transaction() as sess:
            user = UserRepository.get_by_username(self.username)
            sess['user_id'] = user['id']
            sess['username'] = user['username']
            sess['fingerprint_type'] = 'loop'
            sess['ridge_count'] = 18

    def test_predict_form_submission(self):
        form_data = {
            'age': '48',
            'bmi': '27.4',
            'family_history': 'yes',
            'fingerprint_type_hidden': 'loop',
            'ridge_count_hidden': '18',
            'patient_name': 'Test Case John',
            'model_choice': 'Random Forest'
        }
        response = self.app.post('/predict_diabetes', data=form_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Diagnostic Risk Dossier", response.data)
        self.assertIn(b"AI CONTINUOUS RISK ESTIMATE", response.data)
        self.assertIn(b"/ 100", response.data)

if __name__ == '__main__':
    unittest.main()
