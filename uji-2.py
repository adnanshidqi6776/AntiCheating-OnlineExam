from ultralytics import YOLO
import torch
import warnings
import cv2
import time
import screeninfo
import os
import threading
import queue

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# MODEL
# =========================
model = YOLO(r"runs\detect\train\weights\best.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Tidak dapat mengakses kamera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# =========================
# SCREEN
# =========================
screen = screeninfo.get_monitors()[0]
screen_w, screen_h = screen.width, screen.height

# =========================
# FPS
# =========================
target_fps = 20
frame_time = 1 / target_fps

# =========================
# WINDOW
# =========================
window_name = "Deteksi Objek - YOLOv8 (Realtime)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, screen_w, screen_h)
cv2.moveWindow(window_name, 0, 0)

# =========================
# OUTPUT DIR
# =========================
video_dir = "output/video"
frame_dir = "output/frames"
os.makedirs(video_dir, exist_ok=True)
os.makedirs(frame_dir, exist_ok=True)

# =========================
# VIDEO WRITER
# =========================
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(
    f"{video_dir}/hasil_deteksi.mp4",
    fourcc,
    target_fps,
    (screen_w, screen_h)
)

# =========================
# MULTI THREAD SETUP
# =========================
save_queue = queue.Queue(maxsize=100)
stop_event = threading.Event()

def saving_worker():
    img_id = 0
    while not stop_event.is_set() or not save_queue.empty():
        try:
            frame, save_image = save_queue.get(timeout=0.1)

            video_writer.write(frame)

            if save_image:
                filename = f"{frame_dir}/frame_{img_id:06d}.jpg"
                cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                img_id += 1

            save_queue.task_done()
        except queue.Empty:
            continue

save_thread = threading.Thread(target=saving_worker, daemon=True)
save_thread.start()

print("Kamera aktif.")
print("Tekan 'q' untuk keluar | Tekan 'spasi' untuk pause/resume")

# =========================
# MAIN LOOP
# =========================
frame_count = 0
paused = False
last_frame = None

while True:
    start_time = time.time()

    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # =========================
        # YOLO INFERENCE (threshold > 0.5)
        # =========================
        results = model.predict(
            frame,
            conf=0.5,
            imgsz=640,
            verbose=False
        )

        # mirror video
        display_frame = cv2.flip(frame, 1)

        # =========================
        # DRAW BBOX & LABEL
        # =========================
        boxes = results[0].boxes
        names = results[0].names

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0]

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

        annotated_frame = cv2.resize(
            display_frame,
            (screen_w, screen_h),
            interpolation=cv2.INTER_LINEAR
        )

        last_frame = annotated_frame.copy()

        # =========================
        # SAVE
        # =========================
        save_image = (frame_count % 25 == 0)
        if not save_queue.full():
            save_queue.put((annotated_frame.copy(), save_image))

        frame_count += 1

    # =========================
    # DISPLAY
    # =========================
    if last_frame is not None:
        cv2.imshow(window_name, last_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == 32:  # SPASI
        paused = not paused
        print("PAUSE" if paused else "RESUME")

    if not paused:
        elapsed = time.time() - start_time
        sleep_time = frame_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

# =========================
# CLEAN UP
# =========================
stop_event.set()
save_thread.join()

cap.release()
video_writer.release()
cv2.destroyAllWindows()
