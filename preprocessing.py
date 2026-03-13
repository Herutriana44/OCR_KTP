"""
Image Preprocessing untuk KTP
Menggunakan Edge Detection + Contour Detection untuk crop area KTP
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any


def preprocess_ktp_image(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Preprocessing gambar KTP dengan Edge Detection + Contour Detection.
    
    Args:
        image_path: Path ke file gambar
        
    Returns:
        Dict dengan keys: original, preprocessed, edge_detection, contour_detection, cropped
        atau None jika gagal
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        original = img.copy()
        
        # Resize jika terlalu besar (untuk performa)
        max_dim = 1200
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
        # 1. Preprocessing dasar: grayscale + noise reduction
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        preprocessed = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        
        # 2. Edge Detection (Canny)
        edges = cv2.Canny(blurred, 50, 150)
        edge_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 3. Contour Detection
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Gambar contour pada image (hijau untuk semua, biru untuk KTP terpilih)
        contour_vis = img.copy()
        cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 2)
        
        # 4. Cari contour KTP (area terbesar dengan aspect ratio KTP)
        max_area = 0
        best_contour = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > img.shape[0] * img.shape[1] * 0.1:
                x, y, cw, ch = cv2.boundingRect(contour)
                aspect_ratio = cw / float(ch) if ch > 0 else 0
                # KTP Indonesia: 85.6mm x 53.98mm ≈ 1.58
                if 1.2 < aspect_ratio < 2.2 and area > max_area:
                    max_area = area
                    best_contour = contour
        
        # Gambar contour terpilih dengan warna berbeda
        if best_contour is not None:
            cv2.drawContours(contour_vis, [best_contour], -1, (255, 0, 0), 3)
            x, y, cw, ch = cv2.boundingRect(best_contour)
            cv2.rectangle(contour_vis, (x, y), (x + cw, y + ch), (255, 0, 0), 2)
        
        # 5. Crop area KTP
        if best_contour is not None:
            x, y, cw, ch = cv2.boundingRect(best_contour)
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img.shape[1], x + cw + padding)
            y2 = min(img.shape[0], y + ch + padding)
            cropped = img[y1:y2, x1:x2].copy()
        else:
            # Fallback: gunakan full image dengan enhancement
            cropped = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        
        return {
            'original': original,
            'preprocessed': preprocessed,
            'edge_detection': edge_vis,
            'contour_detection': contour_vis,
            'cropped': cropped
        }
        
    except Exception as e:
        print(f"Error preprocessing: {e}")
        return None


def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Enhancement tambahan untuk meningkatkan akurasi OCR.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
