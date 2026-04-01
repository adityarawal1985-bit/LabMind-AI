from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    # Get detected objects
    names = results[0].names
    classes = results[0].boxes.cls.tolist() if results[0].boxes else []

    detected = [names[int(i)] for i in classes]

    # 🔥 SMART SAFETY LOGIC
    alerts = []

    if "person" in detected:

        # Fake goggles detection (using tie)
        if "tie" not in detected:
            alerts.append("⚠️ Goggles NOT detected")

        # Fake gloves detection (using cell phone)
        if "cell phone" not in detected:
            alerts.append("⚠️ Gloves NOT detected")

        if len(alerts) == 0:
            alerts.append("✅ All safety measures OK")

    else:
        alerts.append("No person detected")

    # Show alerts on screen
    y = 40
    for alert in alerts:
        cv2.putText(annotated_frame, alert, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y += 30

    cv2.imshow("LabMind AI", annotated_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()