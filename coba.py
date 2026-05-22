from ultralytics import YOLO
import cv2

# Load model hasil training
model = YOLO("best.pt")

# Ganti sesuai file yang mau dites:
# Bisa gambar: "test.jpg"
# Bisa video: "video.mp4"
# Bisa webcam: 0
source = "coba1.jpg"

results = model.predict(
    source=source,
    conf=0.35,
    imgsz=640,
    save=True,
    show=True
)

print("Deteksi selesai.")
print("Hasil tersimpan di folder runs/detect/predict-2")