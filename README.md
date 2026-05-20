---
title: Sistem Anti Cheating Online Exam
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: server.py
pinned: false
---

# Sistem Anti Cheating Online Exam (Skripsi Project)

This project is an implementation of **YOLOv8n (Ultralytics) + Flask Framework**
to detect cheating behavior during online exams in real-time using webcam monitoring.

This system was developed as an undergraduate thesis project.

---

# Main Features

- Input student identity (Name & NIM)
- Set maximum cheating tolerance
- Real-time object detection using YOLOv8 custom model
- Detect cheating objects:
  - Handphone
  - Book
  - Finger
- Display warning notification when cheating is detected
- Record cheating evidence automatically
- Save cheating evidence as video (`.webm`)
- Save student answers into Microsoft Word (`.docx`)
- Cheating counter & progress bar

---

# Technology Stack

- Python
- Flask
- YOLOv8n (Ultralytics)
- OpenCV
- Bootstrap 5
- HTML/CSS/JavaScript
- PyTorch

---

# Folder Structure

```text
skripsi-yolo/
│
├── static/
│   ├── home.html
│   └── testPage.html
│
├── detected_image/
│
├── answer/
│
├── logs/
│
├── runs/
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt
│
├── server.py
├── requirements.txt
├── README.md
```
