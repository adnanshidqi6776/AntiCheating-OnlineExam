---
title: Sistem Anti Cheating Online Exam
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: server.py
pinned: false
---

# Sistem Anti Cheating Online Exam (Skripsi Project)

Proyek ini adalah implementasi **FastAPI + YOLOv8 (Ultralytics)** untuk deteksi perilaku mencurigakan secara **real-time** melalui **WebSocket**.

Didesain untuk keperluan penelitian/skripsi, aplikasi ini mampu:

- Mendeteksi objek seperti _finger_, _book_, _handphone_ menggunakan model YOLOv8 custom.
- Mengirim hasil deteksi bounding box ke frontend melalui WebSocket.
- Menyimpan frame hasil deteksi secara otomatis.
- Menampilkan notifikasi peringatan bila deteksi melebihi batas tertentu.

---

## Deploy di Hugging Face Spaces

Aplikasi ini siap dijalankan di **Hugging Face Spaces (SDK: FastAPI)**.

### **Langkah-langkah:**

1. Buat Space baru di [Hugging Face Spaces](https://huggingface.co/spaces)
2. Pilih:
   - **SDK** → `FastAPI`
   - **Hardware** → `GPU` (agar YOLO dapat menggunakan CUDA)
   - **Repository Source** → `From GitHub`
3. Hubungkan dengan repo GitHub kamu yang berisi file ini (`server.py`, `requirements.txt`, dan model YOLO).
4. Deploy — Spaces akan otomatis menginstal dependensi dan menjalankan server.

---

## 🧩 Struktur Folder

skripsi-yolo/
├── server.py
├── requirements.txt
├── README.md
└── runs/
└── detect/
└── train/
└── weights/
└── best.pt

---

## ⚙️ Menjalankan di Lokal

### 1. Buat virtual environment (opsional tapi disarankan)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
