"""
Flask Web Application untuk OCR KTP
Flow: Kamera Webcam -> Ambil Gambar -> Preprocessing (OpenCV) -> OCR (PaddleOCR) -> Tampilkan Hasil
"""

import os
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np

from preprocessing import preprocess_ktp_image, enhance_for_ocr
from ocr_processor import run_ocr, extract_ktp_info

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join(os.path.dirname(__file__), 'processed')

# Buat folder jika belum ada
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
    Flow: Simpan -> Preprocess -> OCR -> Return hasil
    """
    if 'file' not in request.files and 'image' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diupload'}), 400
    
    file = request.files.get('file') or request.files.get('image')
    
    if not file or (file.filename == '' and not file.content_length):
        return jsonify({'error': 'File tidak dipilih atau gambar tidak valid'}), 400
    
    # Untuk capture dari webcam, filename bisa kosong - gunakan default
    filename = secure_filename(file.filename) if file.filename else 'capture.jpg'
    if not allowed_file(filename):
        return jsonify({'error': 'Format file tidak didukung. Gunakan PNG, JPG, atau JPEG.'}), 400
    
    try:
        # Simpan file upload (filename sudah di-set di atas)
        unique_id = str(uuid.uuid4())[:8]
        save_filename = f"{unique_id}_{filename}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], save_filename)
        file.save(upload_path)
        
        # 1. PREPROCESSING - Crop dan enhance dengan OpenCV
        processed_img = preprocess_ktp_image(upload_path)
        
        if processed_img is None:
            # Fallback: baca langsung
            processed_img = cv2.imread(upload_path)
        
        if processed_img is None:
            return jsonify({'error': 'Gagal memproses gambar'}), 500
        
        # Enhancement untuk OCR
        enhanced_img = enhance_for_ocr(processed_img)
        
        # Simpan gambar yang sudah diproses (untuk preview)
        processed_filename = f"proc_{unique_id}.jpg"
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        cv2.imwrite(processed_path, enhanced_img)
        
        # 2. OCR - PaddleOCR
        ocr_results, ktp_info = run_ocr(processed_path, use_angle_cls=True)
        
        # Jika OCR dari processed gagal, coba dari original
        if not ktp_info.get('raw_text') and not ktp_info.get('NIK'):
            ocr_results, ktp_info = run_ocr(upload_path, use_angle_cls=True)
        
        # Bersihkan field kosong dari raw untuk response
        display_info = {k: v for k, v in ktp_info.items() 
                       if k not in ('raw_text', 'raw_lines') and v}
        
        # Format raw lines untuk ditampilkan
        raw_lines = ktp_info.get('raw_lines', [])
        
        return jsonify({
            'success': True,
            'ktp_data': display_info,
            'raw_lines': raw_lines,
            'processed_image': f'/processed/{processed_filename}',
            'original_image': f'/uploads/{save_filename}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve file dari folder uploads"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/processed/<filename>')
def serve_processed(filename):
    """Serve file dari folder processed"""
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
