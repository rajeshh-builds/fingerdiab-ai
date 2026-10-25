"""
FingerDiab AI — Computer Vision Biometric Processing Service
Implements a 5-stage image processing pipeline:
1. Quality Assurance Filter (Blur, Contrast, Resolution)
2. Contrast-Limited Adaptive Histogram Equalization (CLAHE) & Denoising
3. Otsu Optimal Adaptive Binarization
4. Zhang-Suen Morphological Skeletonization / Ridge Thinning
5. Minutiae Feature Extraction & Ridge Density Estimation
"""

import cv2
import numpy as np
import base64
import os
from fingerdiab.utils.quality_check import assess_fingerprint_quality

def array_to_base64(img_arr, is_gray=True):
    """Converts a numpy image array into a base64 png data URI string."""
    if is_gray and len(img_arr.shape) == 2:
        _, buffer = cv2.imencode('.png', img_arr)
    else:
        _, buffer = cv2.imencode('.png', img_arr)
    encoded = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

class FingerprintService:
    @staticmethod
    def process_image(image_path):
        """
        Executes complete 5-stage computer vision pipeline on fingerprint image.
        
        Args:
            image_path (str): File path of uploaded image.
            
        Returns:
            dict:
                - success (bool)
                - quality (dict): Quality metrics and scores
                - features (dict): pattern_type, ridge_count, minutiae_count
                - stages (dict): base64 strings of original, equalized, binary, skeleton, minutiae
                - error (str or None)
        """
        if not os.path.exists(image_path):
            return {'success': False, 'error': 'Image file not found on server.', 'quality': None}
            
        img = cv2.imread(image_path)
        if img is None:
            return {'success': False, 'error': 'Failed to decode image file.', 'quality': None}
            
        # 1. Quality Assurance Check
        quality = assess_fingerprint_quality(img)
        if not quality['is_valid']:
            return {
                'success': False,
                'error': quality['error_message'],
                'quality': quality,
                'stages': {}
            }
            
        # Convert to Grayscale and Standardized Size (300x300)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        gray_resized = cv2.resize(gray, (300, 300), interpolation=cv2.INTER_AREA)
        
        # 2. Contrast Enhancement (CLAHE + Bilateral Filtering)
        denoised = cv2.bilateralFilter(gray_resized, d=7, sigmaColor=50, sigmaSpace=50)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        equalized = clahe.apply(denoised)
        
        # 3. Otsu Adaptive Binarization
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 4. Zhang-Suen Morphological Skeletonization (1-pixel width ridges)
        skeleton = np.zeros(binary.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp_bin = binary.copy()
        
        for _ in range(50): # Max iterations
            eroded = cv2.erode(temp_bin, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(temp_bin, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            temp_bin = eroded.copy()
            if cv2.countNonZero(temp_bin) == 0:
                break
                
        # 5. Ridge Counting and Minutiae Detection
        # Sample across central horizontal & vertical cross-sections
        center_y = 150
        h_line = binary[center_y, 50:250]
        h_transitions = np.sum(np.diff(h_line > 0) != 0) // 2
        
        center_x = 150
        v_line = binary[50:250, center_x]
        v_transitions = np.sum(np.diff(v_line > 0) != 0) // 2
        
        raw_ridge_count = int((h_transitions + v_transitions) / 2)
        # Bounded between biological plausible limits [10, 25] matching training dataset
        ridge_count = int(np.clip(raw_ridge_count, 10, 25))
        
        # Minutiae Visualization (Ridge Endings & Bifurcations)
        minutiae_img = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        skel_norm = (skeleton > 0).astype(np.uint8)
        
        endings = 0
        bifurcations = 0
        
        for r in range(1, 299):
            for c in range(1, 299):
                if skel_norm[r, c] == 1:
                    # 8-neighborhood
                    neighbors = skel_norm[r-1:r+2, c-1:c+2]
                    transitions = (
                        abs(int(neighbors[0,0]) - int(neighbors[0,1])) +
                        abs(int(neighbors[0,1]) - int(neighbors[0,2])) +
                        abs(int(neighbors[0,2]) - int(neighbors[1,2])) +
                        abs(int(neighbors[1,2]) - int(neighbors[2,2])) +
                        abs(int(neighbors[2,2]) - int(neighbors[2,1])) +
                        abs(int(neighbors[2,1]) - int(neighbors[2,0])) +
                        abs(int(neighbors[2,0]) - int(neighbors[1,0])) +
                        abs(int(neighbors[1,0]) - int(neighbors[0,0]))
                    ) // 2
                    
                    if transitions == 1: # Ridge Ending
                        endings += 1
                        cv2.circle(minutiae_img, (c, r), 3, (0, 0, 255), 1)
                    elif transitions == 3: # Bifurcation
                        bifurcations += 1
                        cv2.circle(minutiae_img, (c, r), 3, (0, 255, 0), 1)
                        
        # 6. Dermatoglyphic Pattern Classification Heuristic
        # Analyze circular variance & moments of central core
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_circ = 0
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
            if perimeter > 0:
                total_circ = (4 * np.pi * area) / (perimeter * perimeter)
                
        if total_circ > 0.45 or bifurcations > 45:
            pattern_type = 'whorl'
        elif raw_ridge_count < 13:
            pattern_type = 'arch'
        else:
            pattern_type = 'loop'
            
        # Convert all 5 stages to Base64
        stages = {
            'original': array_to_base64(gray_resized, is_gray=True),
            'equalized': array_to_base64(equalized, is_gray=True),
            'binary': array_to_base64(binary, is_gray=True),
            'skeleton': array_to_base64(skeleton, is_gray=True),
            'minutiae': array_to_base64(minutiae_img, is_gray=False)
        }
        
        return {
            'success': True,
            'quality': quality,
            'features': {
                'fingerprint_type': pattern_type,
                'ridge_count': ridge_count,
                'ridge_endings': endings,
                'bifurcations': bifurcations,
                'minutiae_count': endings + bifurcations
            },
            'stages': stages,
            'error': None
        }
