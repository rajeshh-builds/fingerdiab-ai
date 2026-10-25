"""
FingerDiab AI — Database Repositories
Encapsulates all database operations behind parameterized, injection-proof interfaces.
"""

from fingerdiab.database.database import get_db_connection
from fingerdiab.utils.security import hash_password, verify_password

class UserRepository:
    @staticmethod
    def get_by_id(user_id):
        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        conn.close()
        return dict(user) if user else None
        
    @staticmethod
    def get_by_username(username):
        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()
        return dict(user) if user else None
        
    @staticmethod
    def create_user(username, plain_password, email=None, full_name=None, gender=None):
        hashed = hash_password(plain_password)
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password, email, full_name, gender) VALUES (?, ?, ?, ?, ?)",
                (username, hashed, email, full_name, gender)
            )
            conn.commit()
            new_id = cursor.lastrowid
            return True, new_id, None
        except Exception as e:
            conn.rollback()
            if 'UNIQUE' in str(e).upper():
                return False, None, "Username is already taken."
            return False, None, f"Database error: {str(e)}"
        finally:
            conn.close()
            
    @staticmethod
    def verify_credentials(username, password):
        user = UserRepository.get_by_username(username)
        if not user:
            return False, None
        if verify_password(user['password'], password):
            return True, user
        return False, None

class PredictionRepository:
    @staticmethod
    def save_prediction(user_id, patient_name, fp_type, ridge_count, age, bmi, fam_hist,
                        diabetes_risk, risk_score, model_used, confidence, model_version='fingerdiab-rf-v1.0.0', notes=''):
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (
                    user_id, patient_name, fingerprint_type, ridge_count,
                    age, bmi, family_history, diabetes_risk, risk_score,
                    model_used, confidence, model_version, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, patient_name, fp_type, ridge_count,
                age, bmi, fam_hist, diabetes_risk, risk_score,
                model_used, confidence, model_version, notes
            ))
            conn.commit()
            pred_id = cursor.lastrowid
            return pred_id
        finally:
            conn.close()

    @staticmethod
    def get_by_id(pred_id):
        conn = get_db_connection()
        pred = conn.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,)).fetchone()
        conn.close()
        return dict(pred) if pred else None

    @staticmethod
    def get_by_user_id(user_id, limit=50):
        conn = get_db_connection()
        rows = conn.execute(
            "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def get_user_analytics_summary(user_id):
        conn = get_db_connection()
        preds = conn.execute(
            "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at ASC",
            (user_id,)
        ).fetchall()
        conn.close()
        
        preds_list = [dict(p) for p in preds]
        total_count = len(preds_list)
        if total_count == 0:
            return {
                'total_assessments': 0,
                'latest_score': 0.0,
                'avg_score': 0.0,
                'high_risk_count': 0,
                'medium_risk_count': 0,
                'low_risk_count': 0,
                'recent_history': [],
                'score_trend': []
            }
            
        scores = [float(p['risk_score'] or 0.0) for p in preds_list]
        latest_score = scores[-1]
        avg_score = round(sum(scores) / total_count, 1)
        
        high_cnt = sum(1 for p in preds_list if 'high' in str(p['diabetes_risk']).lower())
        med_cnt = sum(1 for p in preds_list if 'medium' in str(p['diabetes_risk']).lower())
        low_cnt = sum(1 for p in preds_list if 'low' in str(p['diabetes_risk']).lower())
        
        return {
            'total_assessments': total_count,
            'latest_score': latest_score,
            'avg_score': avg_score,
            'high_risk_count': high_cnt,
            'medium_risk_count': med_cnt,
            'low_risk_count': low_cnt,
            'recent_history': list(reversed(preds_list[-10:])),
            'score_trend': scores[-10:]
        }
