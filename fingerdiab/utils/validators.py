"""
FingerDiab AI — Input Validation Module
Validates demographic, biometric, and clinical parameters strictly against plausible physiological ranges.
"""

import re

def validate_age(val):
    try:
        age = int(val)
        if 1 <= age <= 120:
            return True, age, None
        return False, None, "Age must be between 1 and 120 years."
    except (ValueError, TypeError):
        return False, None, "Age must be a valid whole number."

def validate_bmi(val):
    try:
        bmi = float(val)
        if 10.0 <= bmi <= 65.0:
            return True, round(bmi, 2), None
        return False, None, "BMI must be between 10.0 and 65.0 kg/m²."
    except (ValueError, TypeError):
        return False, None, "BMI must be a valid decimal number."

def validate_family_history(val):
    if val is None:
        return False, None, "Family history is required (Yes / No)."
    v = str(val).strip().lower()
    if v in ['yes', 'y', 'true', '1']:
        return True, 'yes', None
    if v in ['no', 'n', 'false', '0']:
        return True, 'no', None
    return False, None, "Family history must be 'yes' or 'no'."

def validate_fingerprint_type(val):
    if val is None:
        return False, None, "Fingerprint type is required."
    v = str(val).strip().lower()
    if v in ['arch', 'loop', 'whorl']:
        return True, v, None
    return False, None, "Fingerprint pattern must be one of: Arch, Loop, Whorl."

def validate_ridge_count(val):
    try:
        rc = int(val)
        if 5 <= rc <= 50:
            return True, rc, None
        return False, None, "Ridge count must be between 5 and 50."
    except (ValueError, TypeError):
        return False, None, "Ridge count must be a valid integer."

def validate_username(username):
    if not username or not isinstance(username, str):
        return False, "Username cannot be blank."
    u = username.strip()
    if len(u) < 3 or len(u) > 30:
        return False, "Username must be between 3 and 30 characters."
    if not re.match(r'^[a-zA-Z0-9_.-]+$', u):
        return False, "Username can only contain letters, numbers, dots, hyphens, and underscores."
    return True, None

def validate_email(email):
    if not email or not isinstance(email, str):
        return False, "Email cannot be blank."
    e = email.strip()
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    if not re.match(pattern, e):
        return False, "Invalid email address format."
    return True, None

def validate_clinical_inputs(form_data):
    """
    Validates complete clinical risk intake payload.
    Returns: (is_valid, cleaned_dict, error_messages)
    """
    errors = []
    cleaned = {}
    
    # Age
    ok, age_val, err = validate_age(form_data.get('age'))
    if not ok: errors.append(err)
    else: cleaned['age'] = age_val
    
    # BMI
    ok, bmi_val, err = validate_bmi(form_data.get('bmi'))
    if not ok: errors.append(err)
    else: cleaned['bmi'] = bmi_val
    
    # Family History
    ok, fam_val, err = validate_family_history(form_data.get('family_history'))
    if not ok: errors.append(err)
    else: cleaned['family_history'] = fam_val
    
    # Fingerprint Type
    ok, fp_val, err = validate_fingerprint_type(form_data.get('fingerprint_type'))
    if not ok: errors.append(err)
    else: cleaned['fingerprint_type'] = fp_val
    
    # Ridge Count
    ok, rc_val, err = validate_ridge_count(form_data.get('ridge_count'))
    if not ok: errors.append(err)
    else: cleaned['ridge_count'] = rc_val
    
    return (len(errors) == 0), cleaned, errors
