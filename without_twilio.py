import cv2
from ultralytics import YOLO
import torch



# ----------------- MAIN -----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = YOLO("ambulance2.pt")
model.to(device)

cap = cv2.VideoCapture("video2.mp4")

CONF_THRESHOLD = 0.5
alert_sent = False   # 🔥 prevents multiple SMS

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_frame = cv2.resize(frame, (640, 480))
    results = model(input_frame, conf=CONF_THRESHOLD)[0]

    ambulance_detected = False  # check per frame

    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        label = model.names[int(cls)].lower()

        if label != "ambulance":
            continue

        ambulance_detected = True  # 🚑 detected

        scale_x = frame.shape[1] / 640
        scale_y = frame.shape[0] / 480
        x1, y1, x2, y2 = map(int, box)

        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "AMBULANCE", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # 🚑 SEND SMS ONLY ONCE
    if ambulance_detected and not alert_sent:
        print("Ambulan Detected")
        alert_sent = True

    cv2.imshow("Ambulance Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()