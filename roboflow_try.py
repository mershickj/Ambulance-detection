import cv2
import math
import numpy as np
import requests
import base64
from collections import OrderedDict

# ----------------- Roboflow Config -----------------
ROBOFLOW_URL = "https://detect.roboflow.com/ambulance-jr09u/3"
API_KEY = "ED4wbI13fv76CxaIJde4"
CONFIDENCE = 40  # percent

# ----------------- Utility -----------------
def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0


# ----------------- Roboflow Prediction -----------------
def roboflow_predict(frame):
    # Resize for faster inference (optional but recommended)
    frame = cv2.resize(frame, (640, 480))

    _, buffer = cv2.imencode(".jpg", frame)
    img_str = base64.b64encode(buffer).decode("utf-8")

    response = requests.post(
        f"{ROBOFLOW_URL}?api_key={API_KEY}&confidence={CONFIDENCE}",
        data=img_str,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    return response.json(), frame


# ----------------- Centroid Tracker -----------------
class CentroidTracker:
    def __init__(self, max_distance=60):
        self.next_id = 0
        self.objects = OrderedDict()

    def update(self, detections):
        new_objects = OrderedDict()

        for (cx, cy, label) in detections:
            assigned = False
            for obj_id, (px, py, plabel) in self.objects.items():
                dist = math.hypot(cx - px, cy - py)
                if dist < 60 and label == plabel:
                    new_objects[obj_id] = (cx, cy, label)
                    assigned = True
                    break

            if not assigned:
                new_objects[self.next_id] = (cx, cy, label)
                self.next_id += 1

        self.objects = new_objects
        return self.objects


# ----------------- MAIN PROGRAM -----------------
cap = cv2.VideoCapture("traffic.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

line_y = int(height * 0.65)

LEFT_LANE = np.array([
    (0, int(height * 0.55)),
    (int(width * 0.45), int(height * 0.55)),
    (int(width * 0.35), height),
    (0, height)
])

RIGHT_LANE = np.array([
    (int(width * 0.45), int(height * 0.55)),
    (width, int(height * 0.55)),
    (width, height),
    (int(width * 0.35), height)
])

tracker = CentroidTracker()
counted_ids = set()

classes = ["bike", "car", "auto", "bus", "truck", "ambulance"]
left = {c: 0 for c in classes}
right = {c: 0 for c in classes}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result, resized_frame = roboflow_predict(frame)
    detections = []

    if "predictions" in result:
        for pred in result["predictions"]:
            label = pred["class"]

            x = int(pred["x"])
            y = int(pred["y"])
            w = int(pred["width"])
            h = int(pred["height"])

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            cx = x
            cy = y

            detections.append((cx, cy, label))

            # Draw detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label.upper(), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    objects = tracker.update(detections)

    for obj_id, (cx, cy, label) in objects.items():
        if obj_id not in counted_ids and cy > line_y:
            counted_ids.add(obj_id)

            if point_in_polygon((cx, cy), LEFT_LANE):
                if label in left:
                    left[label] += 1

            elif point_in_polygon((cx, cy), RIGHT_LANE):
                if label in right:
                    right[label] += 1

    # Draw lanes and line
    cv2.polylines(frame, [LEFT_LANE], True, (255, 0, 0), 2)
    cv2.polylines(frame, [RIGHT_LANE], True, (0, 255, 255), 2)
    cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)

    y_text = 30
    cv2.putText(frame,
        f"LEFT  - Bike:{left['bike']} Car:{left['car']} Auto:{left['auto']} Bus:{left['bus']} Truck:{left['truck']} Amb:{left['ambulance']}",
        (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    y_text += 30
    cv2.putText(frame,
        f"RIGHT - Bike:{right['bike']} Car:{right['car']} Auto:{right['auto']} Bus:{right['bus']} Truck:{right['truck']} Amb:{right['ambulance']}",
        (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Lane & Class Wise Vehicle Counting (Roboflow API)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("FINAL COUNTS")
print("LEFT :", left)
print("RIGHT:", right)