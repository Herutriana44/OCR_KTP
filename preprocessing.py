"""
Image Preprocessing untuk KTP
Alur: Image -> Grayscale -> Gaussian Blur -> Canny Edge -> Find Contours 
      -> Largest Rectangle -> Perspective Transform -> Cropped KTP
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Urutkan 4 titik: top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Perspective transform dari 4 titik ke persegi panjang.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


def preprocess_ktp_image(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Preprocessing gambar KTP mengikuti alur:
    Image -> Grayscale -> Gaussian Blur -> Canny Edge -> Find Contours 
    -> Largest Rectangle -> Perspective Transform -> Cropped KTP
    
    Args:
        image_path: Path ke file gambar
        
    Returns:
        Dict dengan keys: original, grayscale, preprocessed (gaussian blur), 
        edge_detection, contour_detection, largest_rectangle, cropped
    """
    try:
        # 1. Image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        original = img.copy()
        
        # Resize jika terlalu besar
        max_dim = 1200
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
        # 2. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayscale_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # 3. Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        preprocessed = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        
        # 4. Canny Edge
        edges = cv2.Canny(blurred, 50, 150)
        edge_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 5. Find Contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        contour_vis = img.copy()
        cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 2)
        
        # 6. Largest Rectangle - cari contour terbesar dengan 4 sudut (persegi panjang)
        max_area = 0
        best_contour = None
        best_approx = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < img.shape[0] * img.shape[1] * 0.1:
                continue
            
            # Approximate contour ke polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Cari yang punya 4 titik (persegi panjang)
            if len(approx) == 4:
                x, y, cw, ch = cv2.boundingRect(approx)
                aspect_ratio = cw / float(ch) if ch > 0 else 0
                if 1.2 < aspect_ratio < 2.2 and area > max_area:
                    max_area = area
                    best_contour = contour
                    best_approx = approx
        
        # Fallback: jika tidak ada 4-titik, cari terbesar dengan aspect ratio KTP
        if best_contour is None:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < img.shape[0] * img.shape[1] * 0.1:
                    continue
                x, y, cw, ch = cv2.boundingRect(contour)
                aspect_ratio = cw / float(ch) if ch > 0 else 0
                if 1.2 < aspect_ratio < 2.2 and area > max_area:
                    max_area = area
                    best_contour = contour
                    best_approx = np.array([
                        [[x, y]], [[x + cw, y]], 
                        [[x + cw, y + ch]], [[x, y + ch]]
                    ])
        
        largest_rect_vis = img.copy()
        if best_contour is not None:
            cv2.drawContours(largest_rect_vis, [best_contour], -1, (255, 0, 0), 3)
            if best_approx is not None:
                cv2.polylines(largest_rect_vis, [best_approx], True, (0, 0, 255), 2)
        
        # 7. Perspective Transform -> 8. Cropped KTP
        if best_approx is not None and len(best_approx) == 4:
            pts = best_approx.reshape(4, 2)
            cropped = four_point_transform(img, pts)
        elif best_contour is not None:
            # Fallback: crop dengan bounding rect
            x, y, cw, ch = cv2.boundingRect(best_contour)
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img.shape[1], x + cw + padding)
            y2 = min(img.shape[0], y + ch + padding)
            cropped = img[y1:y2, x1:x2].copy()
        else:
            cropped = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        
        return {
            'original': original,
            'grayscale': grayscale_vis,
            'preprocessed': preprocessed,
            'edge_detection': edge_vis,
            'contour_detection': contour_vis,
            'largest_rectangle': largest_rect_vis,
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
