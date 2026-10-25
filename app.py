# app.py
"""
FingerDiab AI — Production Application & Health Risk Screening Engine
Biometric Dermatoglyphic & Demographic Diabetes Risk Assessment System.
"""

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, g, flash, jsonify, send_file, make_response
)
import joblib
import pandas as pd
import numpy as np
import os
import re
import sqlite3
import json
import io
import time
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps

# Import core modular engines
from fingerdiab.config import Config
from fingerdiab.database.database import init_db, get_db_connection
from fingerdiab.database.repositories import UserRepository, PredictionRepository
from fingerdiab.services.fingerprint_service import FingerprintService
from fingerdiab.services.risk_service import RiskService
from fingerdiab.services.recommendation_service import RecommendationService
from fingerdiab.ml.inference import get_inference_engine
from fingerdiab.ml.explainability import ExplainabilityEngine
from fingerdiab.utils.validators import (
    validate_clinical_inputs, validate_username, validate_email
)
from fingerdiab.utils.security import (
    is_allowed_file, sanitize_and_validate_upload_path
)
from remedy_engine import get_clinical_recommendations
from fingerdiab.services.recommendation_service import RecommendationService
from fingerdiab.services.sensor_service import BiometricSensorService

app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ensure database schema is ready
init_db()

# Load ML Inference Engine at startup
ml_engine = get_inference_engine()


# ---------------------------------------------------------
# DATABASE CONTEXT MANAGEMENT
# ---------------------------------------------------------

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = get_db_connection()
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            # If AJAX or JSON request, return clean 401 JSON instead of HTML redirect
            if (request.is_json or 
                request.path.startswith('/extract_features') or 
                request.path.startswith('/api/') or 
                request.headers.get('X-Requested-With') == 'XMLHttpRequest' or 
                request.headers.get('Accept', '').find('application/json') != -1):
                return jsonify({
                    'success': False, 
                    'error': 'Session expired or unauthenticated. Please sign in again.', 
                    'session_expired': True
                }), 401
            flash('Please log in to access this feature.', 'info')
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function


# --- Analytical Helpers for Chart & Dashboard ---
def get_chart_data():
    dataset_path = os.path.join(Config.BASE_DIR, 'fingerprint_diabetes_dataset.csv')
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(Config.BASE_DIR, 'data', 'raw', 'fingerprint_diabetes_dataset.csv')
        
    if not os.path.exists(dataset_path):
        return {'stats': {}, 'riskPie': {}, 'fingerprintDoughnut': {}, 'familyHistoryPie': {}, 'avgAgeBar': {}, 'avgBmiBar': {}, 'crosstabStacked': {}, 'ridgeCountLine': {}}
        
    df = pd.read_csv(dataset_path)

    RISK_COLORS = {
        'low': '#10b981',    # Emerald 500
        'medium': '#f59e0b', # Amber 500
        'high': '#ef4444'    # Red 500
    }
    PALETTE = ['#6366f1', '#8b5cf6', '#ec4899', '#06b6d4', '#10b981', '#f59e0b']

    risk_counts = df['Diabetes_Risk'].value_counts()
    risk_labels = [str(x).capitalize() for x in risk_counts.index]
    risk_data = risk_counts.values.tolist()
    risk_colors = [RISK_COLORS.get(str(x).lower(), '#3b82f6') for x in risk_counts.index]

    fp_counts = df['Fingerprint_Type'].value_counts()
    fp_labels = [str(x).capitalize() for x in fp_counts.index]
    fp_data = fp_counts.values.tolist()

    fh_counts = df['Family_History'].value_counts()
    fh_labels = [str(x).capitalize() for x in fh_counts.index]
    fh_data = fh_counts.values.tolist()

    avg_age_risk = df.groupby('Diabetes_Risk')['Age'].mean().to_dict()
    avg_bmi_risk = df.groupby('Diabetes_Risk')['BMI'].mean().to_dict()

    ct = pd.crosstab(df['Fingerprint_Type'], df['Diabetes_Risk'])
    for r in ['low', 'medium', 'high']:
        if r not in ct.columns:
            ct[r] = 0

    crosstab_labels = [str(x).capitalize() for x in ct.index]
    crosstab_datasets = [
        {"label": "Low Risk", "data": ct['low'].tolist(), "backgroundColor": '#10b981'},
        {"label": "Medium Risk", "data": ct['medium'].tolist(), "backgroundColor": '#f59e0b'},
        {"label": "High Risk", "data": ct['high'].tolist(), "backgroundColor": '#ef4444'}
    ]

    avg_ridge_risk = df.groupby('Diabetes_Risk')['Ridge_Count'].mean().to_dict()

    stats_summary = {
        "total_samples": len(df),
        "avg_age": round(float(df['Age'].mean()), 1),
        "avg_bmi": round(float(df['BMI'].mean()), 1),
        "avg_ridge_count": round(float(df['Ridge_Count'].mean()), 1),
        "high_risk_pct": round(float((df['Diabetes_Risk'] == 'high').mean() * 100), 1),
        "med_risk_pct": round(float((df['Diabetes_Risk'] == 'medium').mean() * 100), 1),
        "low_risk_pct": round(float((df['Diabetes_Risk'] == 'low').mean() * 100), 1)
    }

    return {
        "stats": stats_summary,
        "riskPie": {"labels": risk_labels, "data": risk_data, "colors": risk_colors},
        "fingerprintDoughnut": {"labels": fp_labels, "data": fp_data, "colors": PALETTE[:len(fp_labels)]},
        "familyHistoryPie": {"labels": fh_labels, "data": fh_data, "colors": ['#6366f1', '#06b6d4']},
        "avgAgeBar": {"labels": ["Low", "Medium", "High"], "data": [round(avg_age_risk.get('low', 0), 1), round(avg_age_risk.get('medium', 0), 1), round(avg_age_risk.get('high', 0), 1)]},
        "avgBmiBar": {"labels": ["Low", "Medium", "High"], "data": [round(avg_bmi_risk.get('low', 0), 1), round(avg_bmi_risk.get('medium', 0), 1), round(avg_bmi_risk.get('high', 0), 1)]},
        "crosstabStacked": {"labels": crosstab_labels, "datasets": crosstab_datasets},
        "ridgeCountLine": {"labels": ["Low", "Medium", "High"], "data": [round(avg_ridge_risk.get('low', 0), 1), round(avg_ridge_risk.get('medium', 0), 1), round(avg_ridge_risk.get('high', 0), 1)]}
    }


# ==========================================
# PUBLIC PAGES & AUTHENTICATION ROUTES
# ==========================================

@app.route('/favicon.ico')
def favicon():
    return ('', 204)

@app.route('/')
def index_page():
    chart_data = get_chart_data()
    meta = ml_engine.metadata or {}
    benchmarks = meta.get('benchmark_metrics', {})
    return render_template('index.html', stats=chart_data['stats'], metrics=benchmarks)

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/technology')
def technology_page():
    meta = ml_engine.metadata or {}
    return render_template('technology.html', metrics=meta.get('benchmark_metrics', {}))

@app.route('/model_benchmark')
def model_benchmark_page():
    meta = ml_engine.metadata or {}
    return render_template('model_benchmark.html', metrics=meta.get('benchmark_metrics', {}), calibration=meta.get('calibration_metrics', {}))

@app.route('/simulator')
def simulator_page():
    return render_template('simulator.html')

@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        full_name = request.form.get('full_name', '').strip()
        gender = request.form.get('gender', 'Other')
        password = request.form.get('password', '')

        u_ok, u_err = validate_username(username)
        if not u_ok:
            flash(u_err, 'danger')
            return render_template('register.html')

        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'danger')
            return render_template('register.html')

        if email:
            e_ok, e_err = validate_email(email)
            if not e_ok:
                flash(e_err, 'danger')
                return render_template('register.html')

        success, new_id, err = UserRepository.create_user(
            username=username,
            plain_password=password,
            email=email,
            full_name=full_name,
            gender=gender
        )
        if success:
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login_page'))
        else:
            flash(err, 'danger')
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        valid, user = UserRepository.verify_credentials(username, password)
        if valid:
            session.clear()
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['full_name'] = user['full_name'] if user['full_name'] else user['username']
            flash(f"Welcome back, {session.get('full_name')}!", 'success')
            return redirect(url_for('upload_page'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been securely logged out.', 'info')
    return redirect(url_for('index_page'))


# ==========================================
# DASHBOARD & ANALYTICS ROUTES
# ==========================================

@app.route('/dashboard')
@login_required
def dashboard_page():
    chart_data = get_chart_data()
    meta = ml_engine.metadata or {}
    return render_template('dashboard.html', chart_data=chart_data, metrics=meta.get('benchmark_metrics', {}))

@app.route('/api/chart_data')
def api_chart_data():
    return jsonify(get_chart_data())

@app.route('/history')
@login_required
def history_page():
    records = PredictionRepository.get_by_user_id(session['user_id'])
    return render_template('history.html', records=records)

@app.route('/contact', methods=['GET', 'POST'])
def contact_page():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        message = request.form.get('message', '').strip()

        if name and email and message:
            conn = get_db_connection()
            conn.execute(
                "INSERT INTO consultations (name, email, phone, message) VALUES (?, ?, ?, ?)",
                (name, email, phone, message)
            )
            conn.commit()
            conn.close()
            flash('Thank you! Your query has been submitted to our medical research team.', 'success')
            return redirect(url_for('contact_page'))
        else:
            flash('Please fill in all required fields.', 'danger')
    return render_template('contact.html')


# ==========================================
# FINGERPRINT AI PREDICTION PIPELINE
# ==========================================

@app.route('/api/sensor/status', methods=['GET'])
def api_sensor_status():
    sensor_info = BiometricSensorService.detect_hardware_sensor(force_refresh=True)
    return jsonify({
        'success': True,
        'sensor': sensor_info
    })

@app.route('/api/sensor/scan_live', methods=['POST'])
@login_required
def api_sensor_scan_live():
    data = request.get_json(silent=True) or {}
    raw_b64 = data.get('image')
    archetype = data.get('archetype')
    
    res = BiometricSensorService.acquire_live_biometric_features(subject_archetype=archetype, raw_b64_image=raw_b64)
    if not res['success']:
        return jsonify({'success': False, 'error': res.get('error', 'Live capture failed.')})
    
    session['fingerprint_type'] = res['features']['fingerprint_type']
    session['ridge_count'] = res['features']['ridge_count']
    session['fp_confidence'] = res['quality']['quality_score']
    session['terminations_count'] = res['features']['ridge_endings']
    session['bifurcations_count'] = res['features']['bifurcations']
    session['live_hardware_capture'] = True
    session['device_name'] = res.get('device_name', 'Biometric Hardware')
    session.modified = True

    return jsonify(res)

@app.route('/upload_fingerprint', methods=['GET', 'POST'])
@login_required
def upload_page():
    if request.method == 'POST':
        return extract_features_ajax()
    return render_template('upload_fingerprint.html')

# Endpoint alias so url_for('upload_fingerprint') and url_for('upload_page') both resolve
try:
    app.add_url_rule('/upload_fingerprint', endpoint='upload_fingerprint', view_func=upload_page, methods=['GET', 'POST'])
except AssertionError:
    pass

@app.route('/extract_features', methods=['POST'])
@login_required
def extract_features_ajax():
    if 'fingerprint_image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file uploaded.'})

    file = request.files['fingerprint_image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename received.'})

    if not is_allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file format. Please upload BMP, PNG, JPG, or WEBP.'})

    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"{session['user_id']}_{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], save_name)

    try:
        file.save(filepath)

        # Run CV Biometric Pipeline with image quality checking
        res = FingerprintService.process_image(filepath)

        # Clean up disk file after feature extraction to protect biometric privacy
        if os.path.exists(filepath):
            try: os.remove(filepath)
            except OSError: pass

        if not res['success']:
            return jsonify({
                'success': False,
                'error': res.get('error', 'Feature extraction failed.'),
                'quality': res.get('quality')
            })

        # Save extracted features in session
        session['fingerprint_type'] = res['features']['fingerprint_type']
        session['ridge_count'] = res['features']['ridge_count']
        session['fp_confidence'] = res['quality']['quality_score']
        session['terminations_count'] = res['features']['ridge_endings']
        session['bifurcations_count'] = res['features']['bifurcations']
        session['live_hardware_capture'] = False
        session['device_name'] = 'High-Resolution Optical Biometric Scan'
        session.modified = True

        return jsonify({
            'success': True,
            'fingerprint_type': res['features']['fingerprint_type'],
            'ridge_count': res['features']['ridge_count'],
            'confidence': res['quality']['quality_score'],
            'shape_complexity': 'Standard' if res['features']['fingerprint_type'] != 'whorl' else 'High',
            'terminations_count': res['features']['ridge_endings'],
            'bifurcations_count': res['features']['bifurcations'],
            'stages': res['stages']
        })

    except Exception as e:
        if os.path.exists(filepath):
            try: os.remove(filepath)
            except OSError: pass
        return jsonify({'success': False, 'error': f"Processing error: {str(e)}"})

@app.route('/prediction_form', methods=['GET', 'POST'])
@login_required
def show_prediction_form():
    if request.method == 'POST':
        return predict_diabetes()
    fingerprint_type = session.get('fingerprint_type', 'whorl')
    ridge_count = session.get('ridge_count', 18)
    fp_confidence = session.get('fp_confidence', 95.0)
    live_hardware = session.get('live_hardware_capture', False)
    device_name = session.get('device_name', 'Biometric Hardware Sensor')

    models_available = [
        'Auto-Select (AI Metric-Driven Optimizer)',
        'Random Forest',
        'Gradient Boosting',
        'Decision Tree',
        'SVM',
        'KNN',
        'Logistic Regression'
    ]

    return render_template('prediction_form.html',
                           fingerprint_type=fingerprint_type,
                           ridge_count=ridge_count,
                           fp_confidence=fp_confidence,
                           live_hardware=live_hardware,
                           device_name=device_name,
                           models_available=models_available)

@app.route('/predict_diabetes', methods=['POST'])
@login_required
def predict_diabetes():
    fingerprint_type_str = request.form.get('fingerprint_type_hidden', session.get('fingerprint_type', 'arch')).lower()
    ridge_count_str = request.form.get('ridge_count_hidden', str(session.get('ridge_count', 15)))
    patient_name = request.form.get('patient_name', session.get('full_name', session.get('username', 'Anonymous Patient'))).strip()
    selected_model_name = request.form.get('model_choice', 'Auto-Select (AI Metric-Driven Optimizer)')

    form_dict = {
        'age': request.form.get('age'),
        'bmi': request.form.get('bmi'),
        'family_history': request.form.get('family_history'),
        'fingerprint_type': fingerprint_type_str,
        'ridge_count': ridge_count_str
    }

    # Strict Validation
    is_valid, cleaned, errors = validate_clinical_inputs(form_dict)
    if not is_valid:
        for err in errors:
            flash(err, 'danger')
        return redirect(url_for('show_prediction_form'))

    try:
        # Determine model choice (auto vs manual)
        model_arg = 'auto' if 'auto' in selected_model_name.lower() else selected_model_name
        
        # Run ML Inference with calibrated mathematical risk score & dynamic optimization
        pred_res = ml_engine.predict_risk(cleaned, model_choice=model_arg)

        # Calculate local factor attributions
        factor_contributions = ExplainabilityEngine.calculate_factor_attributions(
            cleaned,
            risk_score=pred_res['risk_score'],
            model=ml_engine.primary_model
        )
        session['last_factors'] = factor_contributions
        session['last_selection_reason'] = pred_res.get('model_selection_reason', '')

        # Save record in Database with audit reason in notes
        prediction_id = PredictionRepository.save_prediction(
            user_id=session['user_id'],
            patient_name=patient_name,
            fp_type=cleaned['fingerprint_type'],
            ridge_count=cleaned['ridge_count'],
            age=cleaned['age'],
            bmi=cleaned['bmi'],
            fam_hist=cleaned['family_history'],
            diabetes_risk=pred_res.get('risk_tier', 'medium'),
            risk_score=pred_res['risk_score'],
            model_used=pred_res.get('model_name', selected_model_name),
            confidence=pred_res.get('confidence_score', 95.0),
            model_version=pred_res.get('model_version', 'fingerdiab-rf-v1.2.0'),
            notes=pred_res.get('model_selection_reason', '')
        )

        return redirect(url_for('result_page', pred_id=prediction_id))

    except Exception as e:
        flash(f"Error executing prediction: {e}", 'danger')
        return redirect(url_for('show_prediction_form'))

@app.route('/result')
@login_required
def result_page():
    pred_id = request.args.get('pred_id') or request.args.get('id')
    if pred_id:
        record = PredictionRepository.get_by_id(pred_id)
    else:
        records = PredictionRepository.get_by_user_id(session['user_id'])
        record = records[0] if records else None

    if not record:
        flash("No prediction result found.", 'info')
        return redirect(url_for('upload_page'))

    # Verify authorization
    if record['user_id'] != session['user_id'] and session.get('username') != 'admin':
        flash("Access unauthorized.", 'danger')
        return redirect(url_for('upload_page'))

    clinical_data = get_clinical_recommendations(
        risk_level=record['diabetes_risk'],
        age=record['age'],
        bmi=record['bmi'],
        family_history=record['family_history'],
        fingerprint_type=record['fingerprint_type'],
        ridge_count=record['ridge_count']
    )

    rec_dict = dict(record)
    rec_dict['interpretation'] = RiskService.get_risk_interpretation(record['risk_score'])
    rec_dict['model_probability'] = round(record['risk_score'], 1)
    class_probs_map, _ = RiskService.get_class_probabilities(record['risk_score'])
    rec_dict['class_probs'] = class_probs_map
    rec_dict['selection_reason'] = record.get('notes') or session.get('last_selection_reason', '')
    rec_dict['live_hardware'] = session.get('live_hardware_capture', False)
    rec_dict['device_name'] = session.get('device_name', 'Biometric Hardware Sensor')

    factors = session.get('last_factors')
    if not factors:
        factors = ExplainabilityEngine.calculate_factor_attributions(
            {
                'age': record['age'],
                'bmi': record['bmi'],
                'family_history': record['family_history'],
                'fingerprint_type': record['fingerprint_type'],
                'ridge_count': record['ridge_count']
            },
            risk_score=record['risk_score'],
            model=ml_engine.primary_model
        )

    return render_template('result.html', 
                           record=rec_dict, 
                           clinical=clinical_data,
                           factor_contributions=factors)

@app.route('/remidy')
def remidy():
    risk = request.args.get('diabetes_risk', 'LOW').upper()
    age = int(request.args.get('age', 35))
    bmi = float(request.args.get('bmi', 23.5))
    family_history = request.args.get('family_history', 'no')
    fingerprint_type = request.args.get('fingerprint_type', 'loop')
    ridge_count = int(request.args.get('ridge_count', 18))

    clinical_data = get_clinical_recommendations(
        risk_level=risk, age=age, bmi=bmi,
        family_history=family_history,
        fingerprint_type=fingerprint_type,
        ridge_count=ridge_count
    )

    return render_template('remidy.html',
                           diabetes_risk=risk,
                           clinical=clinical_data,
                           age=age, bmi=bmi,
                           family_history=family_history,
                           fingerprint_type=fingerprint_type,
                           ridge_count=ridge_count)


# ==========================================
# REST API ENDPOINTS
# ==========================================

@app.route('/api/health', methods=['GET'])
@app.route('/api/v1/health', methods=['GET'])
def api_health():
    return jsonify({
        'status': 'healthy' if ml_engine.is_ready else 'degraded',
        'timestamp': datetime.utcnow().isoformat(),
        'model_version': Config.MODEL_VERSION,
        'model_loaded': ml_engine.is_ready
    })

@app.route('/api/auth/register', methods=['POST'])
def api_register():
    data = request.get_json(force=True) or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    email = data.get('email', '').strip()
    full_name = data.get('full_name', '').strip()
    gender = data.get('gender', 'Other')

    u_ok, u_err = validate_username(username)
    if not u_ok: return jsonify({'success': False, 'error': u_err}), 400
    if len(password) < 6: return jsonify({'success': False, 'error': 'Password minimum 6 characters.'}), 400

    success, new_id, err = UserRepository.create_user(username, password, email, full_name, gender)
    if success:
        return jsonify({'success': True, 'user_id': new_id, 'message': 'User registered successfully.'}), 201
    return jsonify({'success': False, 'error': err}), 400

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    data = request.get_json(force=True) or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    valid, user = UserRepository.verify_credentials(username, password)
    if valid:
        session.clear()
        session['user_id'] = user['id']
        session['username'] = user['username']
        return jsonify({
            'success': True,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'full_name': user['full_name']
            }
        })
    return jsonify({'success': False, 'error': 'Invalid credentials.'}), 401

@app.route('/api/predictions/analyze', methods=['POST'])
def api_analyze():
    data = request.get_json(force=True) or {}
    is_valid, cleaned, errors = validate_clinical_inputs(data)
    if not is_valid:
        return jsonify({'success': False, 'errors': errors}), 400

    model_choice = data.get('model_choice') or data.get('model', 'auto')
    try:
        model_arg = 'auto' if 'auto' in str(model_choice).lower() else model_choice
        pred_res = ml_engine.predict_risk(cleaned, model_choice=model_arg)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

    factors = ExplainabilityEngine.calculate_factor_attributions(
        cleaned,
        risk_score=pred_res['risk_score'],
        model=ml_engine.primary_model
    )

    user_id = session.get('user_id')
    pred_id = None
    if user_id:
        pred_id = PredictionRepository.save_prediction(
            user_id=user_id,
            patient_name=data.get('patient_name', 'API User'),
            fp_type=cleaned['fingerprint_type'],
            ridge_count=cleaned['ridge_count'],
            age=cleaned['age'],
            bmi=cleaned['bmi'],
            fam_hist=cleaned['family_history'],
            diabetes_risk=pred_res.get('risk_tier', 'medium'),
            risk_score=pred_res['risk_score'],
            model_used=pred_res.get('model_name', 'Random Forest'),
            confidence=pred_res.get('confidence_score', 95.0),
            model_version=pred_res.get('model_version', 'fingerdiab-rf-v1.2.0'),
            notes=pred_res.get('model_selection_reason', '')
        )

    return jsonify({
        'success': True,
        'prediction_id': pred_id,
        'prediction': {
            'risk_score': pred_res['risk_score'],
            'model_probability': pred_res['model_probability'],
            'risk_tier': pred_res.get('risk_tier', 'medium'),
            'risk_interpretation': pred_res.get('interpretation', 'Intermediate Estimated Risk'),
            'confidence_percent': pred_res.get('confidence_score', 95.0),
            'class_probabilities': pred_res.get('class_probabilities', {}),
            'model_selection_reason': pred_res.get('model_selection_reason', '')
        },
        'feature_attributions': factors,
        'features_used': cleaned,
        'model': {
            'name': pred_res.get('model_name', 'Random Forest Classifier'),
            'version': pred_res.get('model_version', 'fingerdiab-rf-v1.2.0')
        },
        'disclaimer': (
            "This is an AI-based preliminary risk screening estimate and is not a medical diagnosis. "
            "Users should consult a qualified healthcare professional for clinical testing and diagnosis."
        )
    })

@app.route('/api/predict_instant', methods=['POST'])
def api_predict_instant():
    return api_analyze()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    print(f"Starting FingerDiab AI Server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=debug)
