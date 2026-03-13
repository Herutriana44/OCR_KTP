"""
Deteksi KTP berbasis YOLOv5 Instance Segmentation
Flow: Image -> YOLOv5 Inference -> Filter class 'ktp' (max 1) -> Crop -> Return
"""

import os
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2

# Path model - best (1).pt
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best (1).pt")
KTP_CLASS_NAME = "ktp"
MAX_DETECTIONS = 1

_model = None


def _load_model():
    """Load YOLOv5 model (lazy loading)."""
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
    import torch
    _model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=MODEL_PATH,
        force_reload=False,
        trust_repo=True,
    )
    _model.conf = 0.25
    _model.iou = 0.45
    _model.max_det = MAX_DETECTIONS
    return _model


def _get_class_names(model) -> list:
    """Ambil nama class dari model."""
    try:
        return list(model.names.values()) if hasattr(model, "names") else []
    except Exception:
        return []


def _crop_ktp_from_image(
    img: np.ndarray, xyxy: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Crop region KTP dari gambar.
    Prioritas: gunakan mask jika ada (instance seg), else bbox.
    """
    x1, y1, x2, y2 = map(int, xyxy[:4])
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if mask is not None and mask.size > 0:
        try:
            mask_resized = cv2.resize(
                mask.astype(np.uint8), (x2 - x1, y2 - y1)
            )
            roi = img[y1:y2, x1:x2].copy()
            roi[mask_resized == 0] = [255, 255, 255]
            return roi
        except Exception:
            pass
    return img[y1:y2, x1:x2].copy()


def detect_and_crop_ktp(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Deteksi KTP dengan YOLOv5 instance segmentation, crop, dan return.

    Args:
        image_path: Path ke file gambar

    Returns:
        Dict dengan keys: original, detection_vis, cropped, ktp_found
        atau None jika gagal load image
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        original = img.copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        model = _load_model()
        results = model(img_rgb)

        class_names = _get_class_names(model)
        ktp_class_id = None
        for i, name in enumerate(class_names):
            if name.lower() == KTP_CLASS_NAME.lower():
                ktp_class_id = i
                break
        if ktp_class_id is None and len(class_names) > 0:
            ktp_class_id = 0

        preds = results.xyxy[0] if len(results.xyxy) > 0 else None
        if preds is None or preds.numel() == 0:
            return {
                "original": original,
                "detection_vis": original.copy(),
                "cropped": original,
                "ktp_found": False,
            }

        preds_np = preds.cpu().numpy()
        ktp_pred = None
        for row in preds_np:
            cls_id = int(row[5]) if len(row) > 5 else 0
            if ktp_class_id is None or cls_id == ktp_class_id:
                ktp_pred = row
                break
        if ktp_pred is None and len(preds_np) > 0:
            ktp_pred = preds_np[0]

        if ktp_pred is None:
            return {
                "original": original,
                "detection_vis": original.copy(),
                "cropped": original,
                "ktp_found": False,
            }

        xyxy = ktp_pred[:4]
        mask = None
        if hasattr(results, "masks") and results.masks is not None:
            try:
                masks = results.masks.data
                if masks is not None and len(masks) > 0:
                    m = masks[0].cpu().numpy()
                    if m.size > 0:
                        mask = cv2.resize(
                            m, (img.shape[1], img.shape[0])
                        )
                        mask = (mask > 0.5).astype(np.uint8)
            except Exception:
                pass

        cropped = _crop_ktp_from_image(img, xyxy, mask)

        detection_vis = img.copy()
        x1, y1, x2, y2 = map(int, xyxy[:4])
        cv2.rectangle(detection_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            detection_vis,
            KTP_CLASS_NAME,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        return {
            "original": original,
            "detection_vis": detection_vis,
            "cropped": cropped,
            "ktp_found": True,
        }
    except Exception as e:
        print(f"Error ktp_detector: {e}")
        return None
