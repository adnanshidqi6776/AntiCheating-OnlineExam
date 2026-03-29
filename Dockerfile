# Gunakan image Python versi 3.10 yang ringan
FROM python:3.10-slim

# Set folder kerja di dalam container
WORKDIR /app

# Salin semua file dari proyek lokal ke folder kerja container
COPY . /app

# Install semua library dari requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Jalankan server FastAPI menggunakan uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
