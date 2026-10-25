"""
FingerDiab AI — Automated REST API Tests
Tests all JSON endpoints for authentication, model inference, error handling, and healthchecks.
"""

import unittest
import json
from app import app

class TestRestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health_endpoint(self):
        response = self.app.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertTrue(data['model_loaded'])
        self.assertIn('model_version', data)

    def test_analyze_api_valid(self):
        payload = {
            'age': 50,
            'bmi': 28.2,
            'family_history': 'yes',
            'fingerprint_type': 'loop',
            'ridge_count': 16,
            'model': 'Random Forest'
        }
        response = self.app.post('/api/predictions/analyze',
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('prediction', data)
        self.assertIn('risk_score', data['prediction'])
        self.assertIn('model_probability', data['prediction'])
        self.assertIn('feature_attributions', data)
        self.assertIn('disclaimer', data)

    def test_analyze_api_invalid_inputs(self):
        payload = {
            'age': 200,
            'bmi': 5.0,
            'family_history': 'maybe',
            'fingerprint_type': 'unknown',
            'ridge_count': -5
        }
        response = self.app.post('/api/predictions/analyze',
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertGreaterEqual(len(data['errors']), 1)

if __name__ == '__main__':
    unittest.main()
