"""
FingerDiab AI — Risk Score Derivation Service
Transforms calibrated multiclass probabilities into continuous, mathematically grounded
numerical risk scores (5.0% to 94.5%) with clinical factor modulation and realistic continuous bands.
"""

import numpy as np

def calculate_numerical_risk_score(probabilities, class_order=('low', 'medium', 'high'), input_dict=None):
    if isinstance(probabilities, (list, np.ndarray)):
        p_dict = {
            class_order[i]: float(probabilities[i]) for i in range(len(class_order))
        }
    else:
        p_dict = {k.lower(): float(v) for k, v in probabilities.items()}
        
    p_low = p_dict.get('low', 0.0)
    p_med = p_dict.get('medium', 0.0)
    p_high = p_dict.get('high', 0.0)
    
    total = p_low + p_med + p_high
    if total > 0:
        p_low /= total
        p_med /= total
        p_high /= total

    # 1. Base ML model probability component:
    # Low risk probability anchors to 12.0%, Medium to 48.0%, High to 82.0%
    base_ml_score = (p_low * 12.0) + (p_med * 48.0) + (p_high * 82.0)
    
    if input_dict and isinstance(input_dict, dict):
        age = float(input_dict.get('age', 40.0))
        bmi = float(input_dict.get('bmi', 25.0))
        fh_val = 1.0 if str(input_dict.get('family_history', '')).lower() in {'yes', '1', 'true'} else 0.0
        rc = float(input_dict.get('ridge_count', 16.0))
        fp_type = str(input_dict.get('fingerprint_type', 'loop')).lower()
        
        # BMI continuous modulation (Normal 18.5 - 24.9 is ~0; overweight 25-29.9 is +2 to +7; obese 30-39 is +8 to +15)
        if bmi < 18.5:
            bmi_mod = -4.0
        elif 18.5 <= bmi <= 24.9:
            bmi_mod = (bmi - 21.7) * 0.8
        elif 25.0 <= bmi <= 29.9:
            bmi_mod = 3.0 + (bmi - 25.0) * 1.2
        elif 30.0 <= bmi <= 39.9:
            bmi_mod = 9.0 + (bmi - 30.0) * 0.9
        else:
            bmi_mod = 18.0
            
        # Age continuous modulation (Young <25 is -8 to -5; 25-44 is -5 to +2; 45-64 is +4 to +12; 65+ is +13 to +17)
        if age < 25:
            age_mod = -8.0 + (age - 18) * 0.3
        elif 25 <= age < 45:
            age_mod = -5.0 + (age - 25) * 0.4
        elif 45 <= age < 65:
            age_mod = 4.0 + (age - 45) * 0.45
        else:
            age_mod = 13.0 + min(5.0, (age - 65) * 0.25)
            
        # Genetic family history modulation
        fh_mod = 6.0 if fh_val == 1.0 else -4.0
        
        # Dermal ridge count modulation (High count >=18 is protective; low count <=11 is elevated risk)
        rc_mod = 0.0
        if rc >= 20:
            rc_mod = -5.0
        elif rc >= 16:
            rc_mod = -2.0
        elif rc <= 11:
            rc_mod = 5.0
        elif rc <= 14:
            rc_mod = 2.5
            
        # Pattern type modulation
        fp_mod = 2.0 if fp_type in {'whorl', 'arch'} else -2.0
        
        # Blended continuous score
        total_continuous = base_ml_score + (bmi_mod * 0.35) + (age_mod * 0.35) + (fh_mod * 0.30) + (rc_mod * 0.25) + (fp_mod * 0.15)
        risk_score = round(float(np.clip(total_continuous, 5.0, 94.5)), 1)
    else:
        risk_score = round(float(np.clip(base_ml_score, 5.0, 94.5)), 1)
    
    model_probability = round(risk_score / 100.0, 3)
    elevated_probability_pct = round(risk_score, 1)
    
    if risk_score <= 33.0:
        predicted_class = 'low'
        interpretation = "Lower Estimated Risk"
        risk_level_code = "low"
        badge_color = "emerald"
    elif risk_score <= 66.0:
        predicted_class = 'medium'
        interpretation = "Intermediate Estimated Risk"
        risk_level_code = "medium"
        badge_color = "amber"
    else:
        predicted_class = 'high'
        interpretation = "Higher Estimated Risk"
        risk_level_code = "high"
        badge_color = "rose"
        
    class_probs, raw_probs = RiskService.get_class_probabilities(risk_score)
    confidence_score = round(max(class_probs.values()), 1)
        
    return {
        'risk_score': risk_score,
        'model_probability': model_probability,
        'elevated_probability_pct': elevated_probability_pct,
        'predicted_class': predicted_class,
        'risk_tier': risk_level_code,
        'risk_level_code': risk_level_code,
        'interpretation': interpretation,
        'badge_color': badge_color,
        'confidence_score': confidence_score,
        'probabilities': raw_probs,
        'class_probabilities': class_probs
    }

class RiskService:
    @staticmethod
    def calculate_risk_score(probabilities, class_order=('low', 'medium', 'high'), input_dict=None):
        return calculate_numerical_risk_score(probabilities, class_order, input_dict=input_dict)

    @staticmethod
    def get_risk_interpretation(risk_score):
        if risk_score <= 33.0:
            return "Lower Estimated Risk"
        elif risk_score <= 66.0:
            return "Intermediate Estimated Risk"
        else:
            return "Higher Estimated Risk"

    @staticmethod
    def get_class_probabilities(risk_score):
        # Normalized continuous Gaussian probability distribution across clinical risk spectrum
        sigma = 22.0
        d_low = ((risk_score - 15.0) / sigma) ** 2
        d_med = ((risk_score - 50.0) / sigma) ** 2
        d_high = ((risk_score - 85.0) / sigma) ** 2
        
        weights = np.exp(-0.5 * np.array([d_low, d_med, d_high]))
        probs = weights / np.sum(weights)
        raw_probs = [float(probs[0]), float(probs[1]), float(probs[2])]
        
        p_high_pct = round(raw_probs[2] * 100.0, 1)
        p_low_pct = round(raw_probs[0] * 100.0, 1)
        p_med_pct = round(100.0 - (p_high_pct + p_low_pct), 1)
        
        class_probs = {
            'low': p_low_pct,
            'medium': p_med_pct,
            'high': p_high_pct
        }
        return class_probs, raw_probs

RiskScoringService = RiskService
