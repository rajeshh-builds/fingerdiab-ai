"""
FingerDiab AI — Automated Biometric Feature Extraction & Quality Tests
Tests image quality metrics, blur/contrast thresholds, and OpenCV feature extraction pipeline.
"""

import unittest
import os
import cv2
import numpy as np
import tempfile
from fingerdiab.services.fingerprint_service import FingerprintService
from fingerdiab.utils.quality_check import assess_fingerprint_quality

class TestBiometricFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.project_dir = os.path.dirname(cls.test_dir)
        cls.sample_img_path = os.path.join(cls.project_dir, 'Test Images', 'cluster_0_26.BMP')

    def test_valid_sample_image_processing(self):
        if not os.path.exists(self.sample_img_path):
            self.skipTest("Sample test image not present.")
            
        res = FingerprintService.process_image(self.sample_img_path)
        self.assertTrue(res['success'])
        self.assertIn(res['features']['fingerprint_type'], ['arch', 'loop', 'whorl'])
        self.assertGreater(res['features']['ridge_count'], 0)
        self.assertGreaterEqual(res['quality']['quality_score'], 0.0)
        self.assertEqual(len(res['stages']), 5)

    def test_blurry_image_rejection(self):
        # Create heavily blurred image
        blank = np.zeros((300, 300), dtype=np.uint8) + 128
        blurred = cv2.GaussianBlur(blank, (51, 51), 0)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, blurred)
            
        try:
            q_res = assess_fingerprint_quality(temp_path)
            self.assertFalse(q_res['is_valid'])
            self.assertIn("insufficient", q_res['error_message'])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == '__main__':
    unittest.main()
