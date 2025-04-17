from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train3_continue/weights/best.pt")

cap = cv2.VideoCapture(0)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    frame_count += 1
    print(f"📷 Processing frame {frame_count}...")

    # Run inference
    results = model.predict(source=frame, conf=0.4, stream=False)
    result = results[0]

    # Check if any boxes were detected
    if result.boxes is not None and len(result.boxes) > 0:
        print(f"✅ Detected {len(result.boxes)} objects")
    else:
        print("⚠️ No detections")

    # Plot or draw manually
    frame = result.plot()

    # Show annotated frame
    cv2.imshow("YOLOv8 Live", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
