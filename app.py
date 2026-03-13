"""
Flask Web Application untuk OCR KTP
Flow: Kamera -> Ambil Gambar -> YOLOv5 KTP Detection (instance seg) -> Crop -> Preprocessing -> OCR -> Tampilkan Hasil
"""

import os
import uuid
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2

from ktp_detector import detect_and_crop_ktp
from preprocessing import enhance_for_ocr
from ocr_processor import run_ocr

try:
    from config import SHOW_INFERENCE_LOG
except ImportError:
    SHOW_INFERENCE_LOG = True

# Setup logging
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'app.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger('ocr_ktp')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join(os.path.dirname(__file__), 'processed')

# Set False untuk menonaktifkan inference log di web
app.config['ENABLE_INFERENCE_LOG'] = SHOW_INFERENCE_LOG

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Halaman utama - kamera webcam untuk foto KTP"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Endpoint upload dan proses OCR.
    Flow: Simpan -> YOLOv5 KTP Detection (max 1, label ktp) -> Crop -> Enhancement -> OCR -> Return hasil
    """
    if 'file' not in request.files and 'image' not in request.files:
        logger.warning("Upload ditolak: tidak ada file di request")
        return jsonify({'error': 'Tidak ada file yang diupload'}), 400
    
    file = request.files.get('file') or request.files.get('image')
    
    if not file or (file.filename == '' and not file.content_length):
        logger.warning("Upload ditolak: file kosong atau tidak valid")
        return jsonify({'error': 'File tidak dipilih atau gambar tidak valid'}), 400
    
    filename = secure_filename(file.filename) if file.filename else 'capture.jpg'
    if not allowed_file(filename):
        logger.warning("Upload ditolak: format tidak didukung - %s", filename)
        return jsonify({'error': 'Format file tidak didukung. Gunakan PNG, JPG, atau JPEG.'}), 400
    
    unique_id = str(uuid.uuid4())[:8]
    logger.info("Upload mulai: %s (id=%s)", filename, unique_id)
    
    try:
        save_filename = f"{unique_id}_{filename}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], save_filename)
        file.save(upload_path)
        
        # 1. YOLOv5 Instance Segmentation - Deteksi KTP (max 1, label ktp)
        det_result = detect_and_crop_ktp(upload_path)
        
        if det_result is None:
            logger.error("Deteksi KTP gagal: det_result None (upload_path=%s)", upload_path)
            return jsonify({'error': 'Gagal memproses gambar'}), 500
        
        def save_img(img, name):
            path = os.path.join(app.config['PROCESSED_FOLDER'], f"{name}_{unique_id}.jpg")
            cv2.imwrite(path, img)
            return f'/processed/{os.path.basename(path)}'
        
        save_img(det_result['original'], 'original')
        save_img(det_result['detection_vis'], 'yolov5_detection')
        save_img(det_result['cropped'], 'crop')
        
        # 2. Enhancement untuk OCR
        enhanced_img = enhance_for_ocr(det_result['cropped'])
        enhanced_path = os.path.join(app.config['PROCESSED_FOLDER'], f"enhanced_{unique_id}.jpg")
        cv2.imwrite(enhanced_path, enhanced_img)
        
        # 3. OCR - PaddleOCR
        show_log = app.config.get('ENABLE_INFERENCE_LOG', True)
        ocr_results, ktp_info, inference_log, detection_vis = run_ocr(
            enhanced_path, use_angle_cls=True, show_log=show_log
        )
        
        if not ktp_info.get('raw_text') and not ktp_info.get('NIK'):
            ocr_results, ktp_info, inference_log, detection_vis = run_ocr(
                upload_path, use_angle_cls=True, show_log=show_log
            )
        
        if detection_vis is not None:
            det_path = os.path.join(app.config['PROCESSED_FOLDER'], f"detection_{unique_id}.jpg")
            cv2.imwrite(det_path, detection_vis)
            ocr_detection_url = f'/processed/{os.path.basename(det_path)}'
        else:
            ocr_detection_url = f'/processed/crop_{unique_id}.jpg'
        
        display_info = {k: v for k, v in ktp_info.items() 
                       if k not in ('raw_text', 'raw_lines') and v}
        
        logger.info("OCR selesai sukses: id=%s, ktp_found=%s, NIK=%s", 
                    unique_id, det_result.get('ktp_found'), ktp_info.get('NIK', '-'))
        
        return jsonify({
            'success': True,
            'ktp_data': display_info,
            'raw_lines': ktp_info.get('raw_lines', []),
            'inference_log': inference_log if show_log else [],
            'inference_log_enabled': show_log,
            'ktp_found': det_result.get('ktp_found', False),
            'images': {
                'original': f'/processed/original_{unique_id}.jpg',
                'yolov5_detection': f'/processed/yolov5_detection_{unique_id}.jpg',
                'cropped': f'/processed/crop_{unique_id}.jpg',
                'enhanced': f'/processed/enhanced_{unique_id}.jpg',
                'ocr_detection': ocr_detection_url
            }
        })
        
    except Exception as e:
        logger.exception("Error memproses upload id=%s: %s", unique_id, str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/processed/<filename>')
def serve_processed(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


if __name__ == '__main__':
    logger.info("OCR KTP App starting - log file: %s", LOG_FILE)
    app.run(debug=True, host='0.0.0.0', port=5000)
