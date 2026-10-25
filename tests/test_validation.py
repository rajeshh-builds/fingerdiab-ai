"""
FingerDiab AI — Automated Input Validation Tests
Tests demographic, biometric, and clinical boundary limits and rejection of physiological anomalies.
"""

import unittest
from fingerdiab.utils.validators import (
    validate_age, validate_bmi, validate_family_history,
    validate_fingerprint_type, validate_ridge_count, validate_clinical_inputs
)

class TestValidation(unittest.TestCase):
    def test_age_validation(self):
        ok, val, _ = validate_age("45")
        self.assertTrue(ok)
        self.assertEqual(val, 45)
        
        ok, _, err = validate_age("-5")
        self.assertFalse(ok)
        
        ok, _, err = validate_age("150")
        self.assertFalse(ok)
        
        ok, _, err = validate_age("forty")
        self.assertFalse(ok)

    def test_bmi_validation(self):
        ok, val, _ = validate_bmi("24.8")
        self.assertTrue(ok)
        self.assertAlmostEqual(val, 24.8)
        
        ok, _, err = validate_bmi("5.0")
        self.assertFalse(ok)
        
        ok, _, err = validate_bmi("90.0")
        self.assertFalse(ok)

    def test_family_history_validation(self):
        ok, val, _ = validate_family_history("yes")
        self.assertTrue(ok)
        self.assertEqual(val, "yes")
        
        ok, val, _ = validate_family_history("no")
        self.assertTrue(ok)
        self.assertEqual(val, "no")
        
        ok, _, err = validate_family_history("maybe")
        self.assertFalse(ok)

    def test_fingerprint_type_validation(self):
        for fp in ['arch', 'loop', 'whorl', 'ARCH', 'Loop']:
            ok, val, _ = validate_fingerprint_type(fp)
            self.assertTrue(ok)
            self.assertIn(val, ['arch', 'loop', 'whorl'])
            
        ok, _, err = validate_fingerprint_type("delta")
        self.assertFalse(ok)

    def test_composite_clinical_validation(self):
        valid_payload = {
            'age': '52',
            'bmi': '29.4',
            'family_history': 'yes',
            'fingerprint_type': 'whorl',
            'ridge_count': '22'
        }
        ok, cleaned, errors = validate_clinical_inputs(valid_payload)
        self.assertTrue(ok)
        self.assertEqual(len(errors), 0)
        self.assertEqual(cleaned['age'], 52)
        
        invalid_payload = {
            'age': '250',
            'bmi': '4.0',
            'family_history': 'unknown',
            'fingerprint_type': 'circle',
            'ridge_count': '-10'
        }
        ok, _, errors = validate_clinical_inputs(invalid_payload)
        self.assertFalse(ok)
        self.assertGreaterEqual(len(errors), 4)

if __name__ == '__main__':
    unittest.main()
