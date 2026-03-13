"""
Deteksi KTP berbasis YOLOv5 Instance Segmentation
Flow: Image -> YOLOv5 Inference -> Filter class 'ktp' (max 1) -> Crop -> Return
"""

import os
import sys
import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2

logger = logging.getLogger('ocr_ktp.ktp_detector')

# YOLOv5 input size
IMGSZ = 640
STRIDE = 32


def _letterbox(im: np.ndarray, new_shape=(640, 640), stride=32) -> Tuple[np.ndarray, float, Tuple[int, int, int, int]]:
    """
    Resize and pad image to new_shape, preserving aspect ratio (YOLOv5 style).
    Returns: (padded_im, ratio, (pad_left, pad_top, new_w, new_h))
    """
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return im, r, (int(left), int(top), new_unpad[0], new_unpad[1])


def _preprocess_for_yolov5(img: np.ndarray) -> Tuple["torch.Tensor", float, Tuple[int, int]]:
    """
    Convert BGR numpy image to YOLOv5 input tensor (BCHW, float32, 0-1).
    Returns: (tensor, ratio, (pad_left, pad_top)) untuk konversi koordinat.
    """
    import torch
    im = img.copy()
    im, ratio, (pad_left, pad_top, new_w, new_h) = _letterbox(im, (IMGSZ, IMGSZ), STRIDE)
    im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).float() / 255.0
    im = im.unsqueeze(0)  # Add batch dim
    return im, ratio, (pad_left, pad_top)

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


def _run_segment_predict(model, img: np.ndarray, image_path: str) -> Tuple[Any, float, int, int]:
    """
    Run inference. YOLOv5 SegmentationModel tidak AutoShape-compatible.
    Returns: (results_like, ratio, pad_left, pad_top)
    """
    import torch
    im_tensor, ratio, (pad_left, pad_top) = _preprocess_for_yolov5(img)
    # Forward - model dari hub adalah DetectionModel/AutoShape, akses .model untuk raw
    raw = model.model(im_tensor) if hasattr(model, "model") else model(im_tensor)
    if isinstance(raw, tuple):
        raw = raw[0]
    if isinstance(raw, list):
        raw = raw[0]
    # NMS - pakai utils dari yolov5 hub cache
    hub_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "ultralytics_yolov5_master")
    for d in [hub_dir, hub_dir.replace("_master", "_main")]:
        if os.path.exists(d) and d not in sys.path:
            sys.path.insert(0, d)
            break
    try:
        from utils.general import non_max_suppression
        pred = non_max_suppression(
            raw, model.conf, model.iou,
            classes=model.classes if hasattr(model, "classes") else None,
            agnostic=getattr(model, "agnostic", False),
            max_det=getattr(model, "max_det", 1000),
        )[0]
        if pred is None or len(pred) == 0:
            class R:
                xyxy = [torch.zeros(0, 6)]
                masks = None
            return R(), ratio, pad_left, pad_top
        # pred: [x1,y1,x2,y2,conf,cls] atau [x1,y1,x2,y2,conf,cls,mask_coeffs...]
        class R:
            xyxy = [pred[:, :6]]
            masks = None
        return R(), ratio, pad_left, pad_top
    except ImportError as e:
        logger.warning("YOLOv5 utils tidak tersedia: %s", e)
        try:
            r = model(image_path)
            return r, ratio, pad_left, pad_top
        except Exception:
            raise RuntimeError(
                "YOLOv5 SegmentationModel tidak kompatibel. "
                "Pastikan yolov5 ter-install: pip install yolov5"
            ) from e


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

        model = _load_model()
        # YOLOv5 SegmentationModel tidak AutoShape-compatible - preprocessing manual
        results, ratio, pad_left, pad_top = _run_segment_predict(model, img, image_path)

        class_names = _get_class_names(model)
        ktp_class_id = None
        for i, name in enumerate(class_names):
            if name.lower() == KTP_CLASS_NAME.lower():
                ktp_class_id = i
                break
        if ktp_class_id is None and len(class_names) > 0:
            ktp_class_id = 0

        preds = results.xyxy[0] if len(results.xyxy) > 0 else None

        def _letterbox_to_orig(xyxy: np.ndarray) -> np.ndarray:
            """Convert koordinat dari letterbox (640x640) ke gambar asli."""
            out = xyxy.copy()
            out[0] = (out[0] - pad_left) / ratio  # x1
            out[1] = (out[1] - pad_top) / ratio   # y1
            out[2] = (out[2] - pad_left) / ratio  # x2
            out[3] = (out[3] - pad_top) / ratio   # y2
            return out
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

        xyxy = _letterbox_to_orig(ktp_pred[:4].copy())
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
        logger.exception("Error ktp_detector (image_path=%s): %s", image_path, str(e))
        return None
