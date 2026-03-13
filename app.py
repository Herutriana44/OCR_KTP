"""
Flask Web Application untuk OCR KTP
Flow: Kamera -> Ambil Gambar -> Edge+Contour Crop -> Preprocessing -> OCR -> Tampilkan Hasil
"""

import os
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2

from preprocessing import preprocess_ktp_image, enhance_for_ocr
from ocr_processor import run_ocr

try:
    from config import SHOW_INFERENCE_LOG
except ImportError:
    SHOW_INFERENCE_LOG = True

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
    Flow: Simpan -> Edge+Contour Crop -> Preprocess -> OCR -> Return hasil
    """
    if 'file' not in request.files and 'image' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diupload'}), 400
    
    file = request.files.get('file') or request.files.get('image')
    
    if not file or (file.filename == '' and not file.content_length):
        return jsonify({'error': 'File tidak dipilih atau gambar tidak valid'}), 400
    
    filename = secure_filename(file.filename) if file.filename else 'capture.jpg'
    if not allowed_file(filename):
        return jsonify({'error': 'Format file tidak didukung. Gunakan PNG, JPG, atau JPEG.'}), 400
    
    try:
        unique_id = str(uuid.uuid4())[:8]
        save_filename = f"{unique_id}_{filename}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], save_filename)
        file.save(upload_path)
        
        # 1. PREPROCESSING - Edge Detection + Contour Detection
        prep_result = preprocess_ktp_image(upload_path)
        
        if prep_result is None:
            return jsonify({'error': 'Gagal memproses gambar'}), 500
        
        # Simpan semua intermediate images
        def save_img(img, name):
            path = os.path.join(app.config['PROCESSED_FOLDER'], f"{name}_{unique_id}.jpg")
            cv2.imwrite(path, img)
            return f'/processed/{os.path.basename(path)}'
        
        save_img(prep_result['original'], 'original')
        save_img(prep_result['grayscale'], 'grayscale')
        save_img(prep_result['preprocessed'], 'preprocessed')
        save_img(prep_result['edge_detection'], 'edge')
        save_img(prep_result['contour_detection'], 'contour')
        save_img(prep_result['largest_rectangle'], 'largest_rect')
        save_img(prep_result['cropped'], 'crop')
        
        # 2. Enhancement untuk OCR
        enhanced_img = enhance_for_ocr(prep_result['cropped'])
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
            detection_url = f'/processed/{os.path.basename(det_path)}'
        else:
            detection_url = f'/processed/crop_{unique_id}.jpg'
        
        display_info = {k: v for k, v in ktp_info.items() 
                       if k not in ('raw_text', 'raw_lines') and v}
        
        return jsonify({
            'success': True,
            'ktp_data': display_info,
            'raw_lines': ktp_info.get('raw_lines', []),
            'inference_log': inference_log if show_log else [],
            'inference_log_enabled': show_log,
            'images': {
                'original': f'/processed/original_{unique_id}.jpg',
                'grayscale': f'/processed/grayscale_{unique_id}.jpg',
                'preprocessed': f'/processed/preprocessed_{unique_id}.jpg',
                'edge_detection': f'/processed/edge_{unique_id}.jpg',
                'contour_detection': f'/processed/contour_{unique_id}.jpg',
                'largest_rectangle': f'/processed/largest_rect_{unique_id}.jpg',
                'cropped': f'/processed/crop_{unique_id}.jpg',
                'ocr_detection': detection_url
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/processed/<filename>')
def serve_processed(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
