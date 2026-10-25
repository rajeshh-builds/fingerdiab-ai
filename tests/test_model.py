"""
FingerDiab AI — Automated Model Inference & Scoring Tests
Tests deterministic prediction, continuous risk scoring, probability bounds, and model versioning.
"""

import unittest
import numpy as np
from fingerdiab.ml.inference import get_inference_engine
from fingerdiab.services.risk_service import RiskService

class TestModelInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = get_inference_engine()

    def test_engine_initialization(self):
        self.assertTrue(self.engine.is_ready)
        self.assertIsNotNone(self.engine.primary_model)
        self.assertIsNotNone(self.engine.pipeline)
        self.assertIsNotNone(self.engine.metadata)

    def test_deterministic_prediction(self):
        payload = {
            'age': 45,
            'bmi': 28.5,
            'family_history': 'yes',
            'fingerprint_type': 'loop',
            'ridge_count': 18
        }
        res1 = self.engine.predict_risk(payload, model_name='random_forest')
        res2 = self.engine.predict_risk(payload, model_name='random_forest')
        
        self.assertEqual(res1['risk_score'], res2['risk_score'])
        self.assertEqual(res1['risk_score'], res2['risk_score'])
        for p1, p2 in zip(res1['probabilities'], res2['probabilities']):
            self.assertAlmostEqual(p1, p2, places=5)

    def test_risk_score_bounds_and_probabilities(self):
        test_cases = [
            {'age': 20, 'bmi': 19.5, 'family_history': 'no', 'fingerprint_type': 'arch', 'ridge_count': 11},
            {'age': 45, 'bmi': 26.0, 'family_history': 'yes', 'fingerprint_type': 'loop', 'ridge_count': 18},
            {'age': 65, 'bmi': 34.5, 'family_history': 'yes', 'fingerprint_type': 'whorl', 'ridge_count': 24}
        ]
        for c in test_cases:
            res = self.engine.predict_risk(c, model_name='random_forest')
            score = res['risk_score']
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)
            
            # Probability sum check
            probs = res['probabilities']
            prob_sum = sum(probs)
            self.assertAlmostEqual(prob_sum, 1.0, places=4)
            
            # Model probability matches normalized score
            self.assertAlmostEqual(res['model_probability'], round(score / 100.0, 4), places=3)

    def test_all_candidate_models(self):
        models = ['random_forest', 'gradient_boosting', 'logistic_regression', 'decision_tree', 'svm', 'knn']
        payload = {'age': 40, 'bmi': 27.0, 'family_history': 'no', 'fingerprint_type': 'loop', 'ridge_count': 15}
        for m in models:
            res = self.engine.predict_risk(payload, model_name=m)
            self.assertIn('risk_score', res)
            self.assertIn('risk_tier', res)

if __name__ == '__main__':
    unittest.main()
