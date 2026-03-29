import os
import cv2
from ultralytics import YOLO
from tkinter import Tk, filedialog

# ==============================
# KONFIGURASI
# ==============================
MODEL_PATH = r"runs\detect\train\weights\best.pt"
OUTPUT_DIR = "output"

CLASS_MAP = {
    "book": 0,
    "finger": 1,
    "handphone": 2
}

CONF_THRESHOLD = 0.5

# ==============================
# PILIH MODE DETEKSI
# ==============================
print("Pilih kelas yang ingin dideteksi:")
print("1. book")
print("2. finger")
print("3. handphone")

choice = input("Masukkan pilihan (1/2/3): ")

if choice == "1":
    active_class = "book"
elif choice == "2":
    active_class = "finger"
elif choice == "3":
    active_class = "handphone"
else:
    print("Pilihan tidak valid!")
    exit()

ACTIVE_CLASS_ID = CLASS_MAP[active_class]
print(f"\nMode deteksi aktif: {active_class.upper()}")

# ==============================
# PILIH GAMBAR (MULTI SELECT)
# ==============================
root = Tk()
root.withdraw()
root.attributes("-topmost", True)  # 🔑 PENTING agar dialog muncul di depan
root.update()

image_paths = filedialog.askopenfilenames(
    title="Pilih gambar",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

root.destroy()

if not image_paths:
    print("Tidak ada gambar dipilih.")
    exit()

# ==============================
# LOAD MODEL
# ==============================
model = YOLO(MODEL_PATH)

# ==============================
# BUAT FOLDER OUTPUT
# ==============================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# PROSES DETEKSI
# ==============================
for img_path in image_paths:
    print(f"Memproses: {os.path.basename(img_path)}")

    img = cv2.imread(img_path)
    if img is None:
        print(f"Gagal membaca gambar: {img_path}")
        continue

    results = model(img, conf=CONF_THRESHOLD)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # FILTER: hanya kelas aktif
        if cls_id != ACTIVE_CLASS_ID:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = f"{active_class} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # ==============================
    # SIMPAN HASIL
    # ==============================
    filename = os.path.basename(img_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(OUTPUT_DIR, f"{name}_output{ext}")

    cv2.imwrite(output_path, img)
    print(f"Hasil disimpan: {output_path}")

print("\nSelesai")
