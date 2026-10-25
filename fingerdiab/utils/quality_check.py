"""
FingerDiab AI — Fingerprint Image Quality Assurance Module
Performs quantitative checks for blur, contrast, brightness, resolution, and ridge clarity
prior to feeding biometric data into the computer vision and ML pipeline.
"""

import cv2
import numpy as np

def assess_fingerprint_quality(image_input):
    """
    Evaluates image quality metrics on raw uploaded fingerprint.
    
    Args:
        image_input: NumPy array (grayscale or BGR) or file path.
        
    Returns:
        dict:
            - is_valid (bool): True if passes all critical thresholds
            - quality_score (float): 0.0 to 100.0 aggregate quality score
            - metrics (dict): blur, contrast, brightness, dimensions
            - issues (list of str): specific quality warnings or failure causes
            - error_message (str or None): user-facing guidance if rejected
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    else:
        img = image_input
        
    if img is None:
        return {
            'is_valid': False,
            'quality_score': 0.0,
            'metrics': {},
            'issues': ['Image could not be decoded.'],
            'error_message': 'Fingerprint image could not be read or is corrupted.'
        }
        
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        
    h, w = gray.shape
    issues = []
    
    # 1. Dimension Check
    if w < 64 or h < 64:
        issues.append(f'Resolution too low ({w}x{h} px, minimum 64x64 px required).')
        
    # 2. Blur / Sharpness Check (Laplacian Variance)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if laplacian_var < 30.0:
        issues.append(f'Image is too blurry (Laplacian variance {laplacian_var:.1f} < 30.0).')
        
    # 3. Brightness / Illumination Check
    mean_val = float(np.mean(gray))
    if mean_val < 20.0:
        issues.append('Image is severely underexposed/too dark.')
    elif mean_val > 245.0:
        issues.append('Image is overexposed/washed out.')
        
    # 4. Contrast Check (Standard Deviation of Intensity)
    std_val = float(np.std(gray))
    if std_val < 15.0:
        issues.append(f'Insufficient ridge contrast (intensity standard deviation {std_val:.1f} < 15.0).')
        
    # 5. Aggregate Quality Score (0 - 100)
    # Scaled components: sharpness (40%), contrast (30%), illumination (30%)
    sharp_score = min(100.0, (laplacian_var / 300.0) * 100.0)
    contrast_score = min(100.0, (std_val / 60.0) * 100.0)
    bright_score = 100.0 - (abs(mean_val - 128.0) / 128.0) * 100.0
    
    quality_score = round(0.4 * sharp_score + 0.3 * contrast_score + 0.3 * bright_score, 1)
    
    is_valid = (len(issues) == 0)
    error_message = None
    if not is_valid:
        error_message = (
            "Fingerprint image quality is insufficient for reliable analysis. "
            "Please upload a clearer, well-lit fingerprint image. " + " ".join(issues)
        )
        
    return {
        'is_valid': is_valid,
        'quality_score': max(5.0, quality_score),
        'metrics': {
            'width': w,
            'height': h,
            'sharpness_variance': round(laplacian_var, 2),
            'contrast_std': round(std_val, 2),
            'brightness_mean': round(mean_val, 2)
        },
        'issues': issues,
        'error_message': error_message
    }
