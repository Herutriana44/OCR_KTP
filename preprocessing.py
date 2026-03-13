"""
Image Preprocessing untuk KTP
Menggunakan OpenCV untuk segmentasi dan crop area KTP
"""

import cv2
import numpy as np
from typing import Optional, Tuple


def preprocess_ktp_image(image_path: str) -> Optional[np.ndarray]:
    """
    Preprocessing gambar KTP dengan OpenCV.
    - Load image
    - Konversi ke grayscale
    - Noise reduction
    - Edge detection
    - Find contours untuk deteksi area KTP
    - Crop area terbesar (asumsi KTP adalah objek terbesar)
    
    Args:
        image_path: Path ke file gambar
        
    Returns:
        Gambar yang sudah di-crop atau None jika gagal
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Resize jika terlalu besar (untuk performa)
        max_dim = 1200
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            
        # Konversi ke grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Noise reduction dengan bilateral filter
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive threshold untuk binerisasi
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations untuk membersihkan noise
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Cari contour dengan area terbesar (asumsi KTP)
        max_area = 0
        best_contour = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter: area minimal 10% dari total image
            if area > img.shape[0] * img.shape[1] * 0.1:
                # Cek aspect ratio (KTP biasanya landscape, ratio ~1.5-1.6)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h) if h > 0 else 0
                # KTP Indonesia: 85.6mm x 53.98mm ≈ 1.58
                if 1.2 < aspect_ratio < 2.2 and area > max_area:
                    max_area = area
                    best_contour = contour
        
        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            # Tambah padding kecil
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img.shape[1], x + w + padding)
            y2 = min(img.shape[0], y + h + padding)
            cropped = img[y1:y2, x1:x2]
            return cropped
        
        # Jika tidak menemukan contour yang cocok, return image dengan preprocessing dasar
        # (enhance contrast untuk OCR yang lebih baik)
        enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        return enhanced
        
    except Exception as e:
        print(f"Error preprocessing: {e}")
        return None


def preprocess_from_array(image_array: np.ndarray) -> Optional[np.ndarray]:
    """
    Preprocessing dari numpy array (untuk upload dari web).
    Simpan ke temp file dulu lalu proses.
    """
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        cv2.imwrite(tmp.name, image_array)
        result = preprocess_ktp_image(tmp.name)
        os.unlink(tmp.name)
    return result


def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Enhancement tambahan untuk meningkatkan akurasi OCR.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # CLAHE untuk enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # Convert back to BGR for PaddleOCR
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
