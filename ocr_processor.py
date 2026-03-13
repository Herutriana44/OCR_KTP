"""
OCR Processor menggunakan PaddleOCR
Deteksi teks dan layout dari gambar KTP
"""

import re
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

try:
    from config import SHOW_INFERENCE_LOG
except ImportError:
    # Set False untuk menonaktifkan inference log di web
    SHOW_INFERENCE_LOG = True


def extract_ktp_info(ocr_results: List) -> Dict[str, str]:
    """
    Parse hasil OCR PaddleOCR menjadi field-field KTP Indonesia.
    
    Format KTP Indonesia:
    - NIK: 16 digit
    - Nama, Tempat/Tgl Lahir, Jenis Kelamin, Alamat, dll
    
    Args:
        ocr_results: Hasil dari PaddleOCR (list of [box, (text, confidence)])
        
    Returns:
        Dictionary dengan field KTP
    """
    # Gabungkan semua teks dengan posisi
    all_texts = []
    for line in ocr_results:
        if line and len(line) >= 2:
            box = line[0]
            text_info = line[1]
            text = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
            # Posisi Y untuk sorting (top to bottom)
            y_center = np.mean([p[1] for p in box])
            all_texts.append((y_center, text.strip()))
    
    # Sort by Y position
    all_texts.sort(key=lambda x: x[0])
    full_text = " ".join([t[1] for t in all_texts])
    lines = [t[1] for t in all_texts]
    
    result = {
        "NIK": "",
        "Nama": "",
        "Tempat Lahir": "",
        "Tanggal Lahir": "",
        "Jenis Kelamin": "",
        "Alamat": "",
        "RT/RW": "",
        "Kel/Desa": "",
        "Kecamatan": "",
        "Agama": "",
        "Status Perkawinan": "",
        "Pekerjaan": "",
        "Kewarganegaraan": "",
        "Berlaku Hingga": "",
        "raw_text": full_text,
        "raw_lines": lines
    }
    
    # Extract NIK (16 digit)
    nik_match = re.search(r'\b(\d{16})\b', full_text)
    if nik_match:
        result["NIK"] = nik_match.group(1)
    
    # Keywords untuk field mapping
    field_keywords = {
        "Nama": ["NAMA", "Nama"],
        "Tempat Lahir": ["TEMPAT", "TGL", "LAHIR", "Tempat", "Lahir"],
        "Tanggal Lahir": ["TGL", "LAHIR", "Lahir"],
        "Jenis Kelamin": ["JENIS", "KELAMIN", "LAKI-LAKI", "PEREMPUAN", "Laki-Laki", "Perempuan"],
        "Alamat": ["ALAMAT", "Alamat"],
        "RT/RW": ["RT/RW", "RT", "RW"],
        "Kel/Desa": ["KEL/DESA", "Kel/Desa", "Desa", "Kelurahan"],
        "Kecamatan": ["KECAMATAN", "Kecamatan"],
        "Agama": ["AGAMA", "Agama"],
        "Status Perkawinan": ["STATUS", "PERKAWINAN", "KAWIN", "Belum", "Kawin", "Cerai"],
        "Pekerjaan": ["PEKERJAAN", "Pekerjaan"],
        "Kewarganegaraan": ["KEWARGANEGARAAN", "WNI", "WNA", "Kewarganegaraan"],
        "Berlaku Hingga": ["BERLAKU", "HINGGA", "SEUMUR HIDUP", "Berlaku", "Seumur"]
    }
    
    # Parse berdasarkan pola label: value
    for i, line in enumerate(lines):
        line_upper = line.upper()
        
        # Nama - biasanya setelah "NAMA"
        if "NAMA" in line_upper and len(line) > 5:
            # Format: "NAMA NAMA_LENGKAP" atau "Nama NAMA_LENGKAP"
            parts = line.split(maxsplit=1)
            if len(parts) >= 2 and parts[0].upper() == "NAMA":
                result["Nama"] = parts[1].strip()
            elif not result["Nama"]:
                result["Nama"] = line.replace("NAMA", "").replace("Nama", "").strip()
        
        # NIK
        if "NIK" in line_upper:
            nik_in_line = re.search(r'\d{16}', line)
            if nik_in_line:
                result["NIK"] = nik_in_line.group()
        
        # Tempat/Tgl Lahir
        if "LAHIR" in line_upper or "TEMPAT" in line_upper:
            # Format: "TEMPAT LAHIR Kota, DD-MM-YYYY"
            clean = re.sub(r'^(TEMPAT|TGL|LAHIR|Tempat|Tgl)\s*[:.]?\s*', '', line, flags=re.I).strip()
            if re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean):
                date_match = re.search(r'(\d{2}[-/]\d{2}[-/]\d{4})', clean)
                if date_match:
                    result["Tanggal Lahir"] = date_match.group(1)
                    result["Tempat Lahir"] = re.sub(r'\d{2}[-/]\d{2}[-/]\d{4}', '', clean).strip(' ,-')
            elif clean and not result["Tempat Lahir"]:
                result["Tempat Lahir"] = clean
        
        # Jenis Kelamin
        if "LAKI-LAKI" in line_upper or "LAKI LAKI" in line_upper:
            result["Jenis Kelamin"] = "LAKI-LAKI"
        elif "PEREMPUAN" in line_upper:
            result["Jenis Kelamin"] = "PEREMPUAN"
        
        # Alamat
        if "ALAMAT" in line_upper and len(line) > 7:
            result["Alamat"] = line.replace("ALAMAT", "").replace("Alamat", "").strip(' :')
        
        # RT/RW
        if "RT" in line_upper and "RW" in line_upper:
            rt_rw = re.search(r'(\d{3})[/\s]*(\d{3})', line)
            if rt_rw:
                result["RT/RW"] = f"{rt_rw.group(1)}/{rt_rw.group(2)}"
        
        # Kel/Desa
        if "KEL" in line_upper and "DESA" in line_upper:
            result["Kel/Desa"] = re.sub(r'^(KEL/DESA|Kel/Desa)\s*[:.]?\s*', '', line, flags=re.I).strip()
        
        # Kecamatan
        if "KECAMATAN" in line_upper and len(line) > 10:
            result["Kecamatan"] = line.replace("KECAMATAN", "").replace("Kecamatan", "").strip(' :')
        
        # Agama
        if "AGAMA" in line_upper and len(line) > 6:
            result["Agama"] = line.replace("AGAMA", "").replace("Agama", "").strip(' :')
        
        # Status Perkawinan
        if "KAWIN" in line_upper or "PERKAWINAN" in line_upper:
            if "BELUM" in line_upper:
                result["Status Perkawinan"] = "Belum Kawin"
            elif "KAWIN" in line_upper:
                result["Status Perkawinan"] = "Kawin"
            elif "CERAI" in line_upper:
                result["Status Perkawinan"] = "Cerai"
            elif "JANDA" in line_upper or "DUDA" in line_upper:
                result["Status Perkawinan"] = "Cerai Mati"
            else:
                result["Status Perkawinan"] = re.sub(r'^(STATUS|PERKAWINAN)\s*[:.]?\s*', '', line, flags=re.I).strip()
        
        # Pekerjaan
        if "PEKERJAAN" in line_upper and len(line) > 10:
            result["Pekerjaan"] = line.replace("PEKERJAAN", "").replace("Pekerjaan", "").strip(' :')
        
        # Kewarganegaraan
        if "KEWARGANEGARAAN" in line_upper or "WNI" in line_upper or "WNA" in line_upper:
            result["Kewarganegaraan"] = "WNI" if "WNI" in line_upper else line.replace("KEWARGANEGARAAN", "").strip(' :')
        
        # Berlaku Hingga
        if "BERLAKU" in line_upper or "SEUMUR" in line_upper:
            if "SEUMUR" in line_upper or "HIDUP" in line_upper:
                result["Berlaku Hingga"] = "SEUMUR HIDUP"
            else:
                date_match = re.search(r'(\d{2}[-/]\d{2}[-/]\d{4})', line)
                if date_match:
                    result["Berlaku Hingga"] = date_match.group(1)
    
    # Fallback: cari di full text untuk field yang kosong
    if not result["Nama"] and "NAMA" in full_text:
        # Coba ekstrak nama antara NAMA dan field berikutnya
        pass
    
    return result


def draw_ocr_boxes(image_path: str, ocr_results: List) -> Optional[np.ndarray]:
    """
    Gambar bounding box dari hasil OCR pada image.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        for item in ocr_results:
            if item and len(item) >= 2:
                box = np.array(item[0]).astype(np.int32)
                text_info = item[1]
                text = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
                conf = text_info[1] if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 0
                
                cv2.polylines(img, [box], True, (0, 255, 0), 2)
                x_min = int(min(p[0] for p in box))
                y_min = int(min(p[1] for p in box))
                label = (text[:25] + '...') if len(text) > 25 else text
                cv2.putText(img, label, (x_min, max(15, y_min - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return img
    except Exception:
        return None


def run_ocr(image_path: str, use_angle_cls: bool = True, 
            show_log: Optional[bool] = None) -> Tuple[List, Dict[str, str], List[str], Optional[np.ndarray]]:
    """
    Jalankan PaddleOCR pada gambar KTP.
    
    Args:
        image_path: Path ke file gambar
        use_angle_cls: Gunakan angle classification (untuk teks miring)
        show_log: Override SHOW_INFERENCE_LOG. None = gunakan default
        
    Returns:
        Tuple (raw_ocr_results, parsed_ktp_info, inference_log, detection_vis_image)
    """
    log_enabled = show_log if show_log is not None else SHOW_INFERENCE_LOG
    inference_log = []
    
    def log(msg: str):
        if log_enabled:
            inference_log.append(msg)
    
    try:
        from paddleocr import PaddleOCR
        
        log("[1/4] Inisialisasi PaddleOCR...")
        ocr = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang='en',
            use_gpu=False,
            show_log=False
        )
        
        log("[2/4] Menjalankan deteksi teks...")
        result = ocr.ocr(image_path, cls=use_angle_cls)
        
        if result is None or (isinstance(result, list) and len(result) == 0):
            log("[3/4] Tidak ada teks terdeteksi")
            return [], {}, inference_log, None
        
        ocr_lines = []
        if isinstance(result, list):
            for page in result:
                if page is not None:
                    for item in page:
                        if item and len(item) >= 2:
                            ocr_lines.append(item)
        
        log(f"[3/4] Terdeteksi {len(ocr_lines)} region teks")
        
        log("[4/4] Parsing field KTP...")
        parsed = extract_ktp_info(ocr_lines)
        
        log(f"[Selesai] NIK: {parsed.get('NIK', '-')} | Nama: {parsed.get('Nama', '-')[:30]}...")
        
        detection_vis = draw_ocr_boxes(image_path, ocr_lines)
        
        return ocr_lines, parsed, inference_log, detection_vis
        
    except ImportError as e:
        log(f"[Error] PaddleOCR not installed: {e}")
        return [], {}, inference_log, None
    except Exception as e:
        log(f"[Error] OCR Error: {e}")
        return [], {}, inference_log, None
