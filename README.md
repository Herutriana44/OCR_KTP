# OCR KTP Indonesia

Aplikasi web Flask untuk ekstraksi informasi dari foto KTP menggunakan PaddleOCR dan OpenCV.

## Flow Program

1. **Upload Foto KTP** - User mengupload foto KTP melalui web interface
2. **Image Preprocessing** - OpenCV melakukan segmentasi dan crop area KTP
3. **OCR** - PaddleOCR mendeteksi teks dan layout
4. **Tampilkan Hasil** - Informasi seperti NIK, Nama, Alamat, dll ditampilkan di web

## Instalasi

```bash
# Buat virtual environment (opsional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

**Catatan:** PaddlePaddle dan PaddleOCR membutuhkan waktu instalasi. Untuk GPU, ganti `paddlepaddle` dengan `paddlepaddle-gpu` di requirements.txt.

## Menjalankan Aplikasi

```bash
python3 app.py
# atau: python app.py
```

Buka browser ke: http://localhost:5000

## Struktur Project

```
OCR_KTP/
├── app.py              # Flask application
├── preprocessing.py   # OpenCV image preprocessing
├── ocr_processor.py    # PaddleOCR & KTP parsing
├── requirements.txt
├── templates/
│   ├── base.html
│   └── index.html
├── uploads/            # File upload (auto-created)
└── processed/          # Gambar hasil preprocessing (auto-created)
```

## Field KTP yang Dideteksi

- NIK
- Nama
- Tempat Lahir
- Tanggal Lahir
- Jenis Kelamin
- Alamat
- RT/RW
- Kel/Desa
- Kecamatan
- Agama
- Status Perkawinan
- Pekerjaan
- Kewarganegaraan
- Berlaku Hingga

## Tips

- Gunakan foto KTP dengan pencahayaan baik
- Pastikan KTP terlihat jelas dan tidak blur
- Format KTP Indonesia standar akan memberikan hasil terbaik
