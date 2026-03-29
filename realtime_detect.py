from ultralytics import YOLO
import torch
import warnings
import cv2
import time
import screeninfo  # untuk ambil resolusi layar otomatis

warnings.filterwarnings("ignore", category=FutureWarning)

# Inisialisasi model
model = YOLO(r"runs\detect\train\weights\best.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Buka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Tidak dapat mengakses kamera.")
    exit()

# Set resolusi tinggi kamera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Ambil ukuran layar monitor utama
screen = screeninfo.get_monitors()[0]
screen_w, screen_h = screen.width, screen.height

print("Kamera aktif. Tekan 'q' untuk keluar.")

# Target FPS
target_fps = 30
frame_time = 1 / target_fps

# Buat jendela biasa tapi besar (tidak fullscreen)
window_name = "Deteksi Objek - YOLOv8 (Realtime)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, screen_w, screen_h)
cv2.moveWindow(window_name, 0, 0)  # Posisikan di pojok kiri atas layar

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # =========================
    # YOLO INFERENCE (FRAME ASLI)
    # =========================
    results = model.predict(
        frame,
        conf=0.5,
        imgsz=640,
        verbose=False
    )

    # =========================
    # MIRROR VIDEO
    # =========================
    display_frame = cv2.flip(frame, 1)

    # =========================
    # DRAW BBOX & LABEL (MANUAL)
    # =========================
    boxes = results[0].boxes
    names = results[0].names

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0]

            # transform koordinat X (mirror)
            x1_m = w - int(x2)
            x2_m = w - int(x1)
            y1 = int(y1)
            y2 = int(y2)

            label = f"{names[cls_id]} {conf:.2f}"

            cv2.rectangle(
                display_frame,
                (x1_m, y1),
                (x2_m, y2),
                (0, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                label,
                (x1_m, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    # resize ke layar
    annotated_frame = cv2.resize(
        display_frame,
        (screen_w, screen_h),
        interpolation=cv2.INTER_LINEAR
    )

    # Tampilkan hasil
    cv2.imshow(window_name, annotated_frame)

    # Stabilkan FPS
    elapsed_time = time.time() - start_time
    wait_time = max(1, int((frame_time - elapsed_time) * 1000))
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
