"""
FingerDiab AI — Clinical Lifestyle & Preventive Guidance Service
Provides evidence-informed, responsible lifestyle protocols tailored to model risk tiers.
Explicitly avoids drug dosing and emphasizes qualified medical consultation.
"""

class RecommendationService:
    @staticmethod
    def get_recommendations(risk_tier, risk_score, patient_factors=None):
        """
        Generates structured general wellness and screening guidance.
        
        Args:
            risk_tier (str): 'Lower Estimated Risk', 'Intermediate Estimated Risk', 'Higher Estimated Risk'
            risk_score (float): 0.0 - 100.0
            patient_factors (dict): optional dict containing age, bmi, family_history
            
        Returns:
            dict with dietary, physical activity, monitoring, and disclaimer guidance.
        """
        bmi = (patient_factors or {}).get('bmi', 24.0)
        age = (patient_factors or {}).get('age', 30)
        fam = (patient_factors or {}).get('family_history', 'no')
        
        disclaimer = (
            "Medical Disclaimer: FingerDiab AI provides preliminary risk estimates based on biometric "
            "and demographic research models. This output is NOT a medical diagnosis, clinical prognosis, "
            "or prescription. Always consult a licensed healthcare practitioner or endocrinologist for "
            "diagnostic evaluation including venous fasting blood glucose and HbA1c testing."
        )
        
        if 'Higher' in risk_tier or risk_score >= 66.0:
            urgency = 'Priority Clinical Consultation Recommended'
            diet = [
                'Prioritize low-glycemic index (GI < 55) complex carbohydrates: whole oats, legumes, non-starchy vegetables.',
                'Eliminate sugar-sweetened beverages, refined carbohydrates, and ultra-processed snacks.',
                'Ensure 30-35g of soluble & insoluble dietary fiber daily to stabilize postprandial glucose excursions.',
                'Practice portion moderation with lean proteins (tofu, legumes, skinless poultry, fish) and healthy fats (olive oil, walnuts).'
            ]
            activity = [
                'Aim for at least 150-180 minutes of moderate-intensity aerobic exercise (e.g., brisk walking, cycling) per week.',
                'Incorporate 2-3 sessions of progressive resistance training weekly to enhance skeletal muscle insulin sensitivity.',
                'Engage in a 10-15 minute gentle walk immediately following major meals to attenuate glucose spikes.'
            ]
            monitoring = [
                'Schedule a formal laboratory assessment: Fasting Plasma Glucose (FPG) and Glycated Hemoglobin (HbA1c).',
                'Track resting blood pressure and lipid profile (triglycerides, HDL, LDL) every 3-6 months.',
                'Consider biometric self-monitoring of weight and waist-to-height ratio weekly.'
            ]
        elif 'Intermediate' in risk_tier or risk_score >= 33.0:
            urgency = 'Routine Preventive Follow-up Advised'
            diet = [
                'Focus on Mediterranean-style dietary patterns: high fiber, unsaturated fatty acids, and vibrant antioxidant-rich produce.',
                'Replace refined grains (white rice, white flour) with whole grains (quinoa, barley, brown rice).',
                'Maintain consistent meal timing to avoid extreme blood sugar fluctuations.',
                'Hydrate adequately with 2.5 to 3.0 liters of pure water daily; limit liquid calories.'
            ]
            activity = [
                'Perform 150 minutes of moderate cardiovascular activity weekly (e.g., brisk walking, swimming, light jogging).',
                'Include 2 days of bodyweight or resistance exercises targeting major muscle groups.',
                'Break up prolonged sedentary sitting with 2-3 minutes of active standing or stretching every hour.'
            ]
            monitoring = [
                'Undergo annual routine metabolic health screening including HbA1c and fasting blood glucose.',
                'Monitor body mass index (BMI) and abdominal circumference on a monthly basis.'
            ]
        else: # Lower Estimated Risk
            urgency = 'Maintenance & General Wellness'
            diet = [
                'Maintain a balanced, nutrient-dense diet rich in seasonal vegetables, whole grains, seeds, and quality proteins.',
                'Limit added sugars to less than 25 grams per day.',
                'Maintain mindful eating habits and adequate hydration.'
            ]
            activity = [
                'Maintain an active lifestyle with at least 150 minutes of varied physical activities weekly.',
                'Combine cardiovascular endurance with flexibility, balance, and core strength training.'
            ]
            monitoring = [
                'Continue standard annual preventive health checkups and maintain active lifestyle habits.'
            ]
            
        personalized_notes = []
        if bmi >= 25.0:
            personalized_notes.append(f"Elevated BMI ({bmi}): Gradual 5-7% weight reduction is clinically shown to significantly enhance glycemic control.")
        if fam.lower() in ['yes', 'true', '1']:
            personalized_notes.append("Positive Family History: Genetic predisposition highlights the benefit of proactive metabolic health surveillance.")
            
        return {
            'urgency_level': urgency,
            'dietary_guidelines': diet,
            'physical_activity': activity,
            'monitoring_schedule': monitoring,
            'personalized_notes': personalized_notes,
            'medical_disclaimer': disclaimer
        }
