from ultralytics import YOLO

class YOLODetector:

    def __init__(self, model="yolov8n.pt"):
        self.model = YOLO(model)

        self.allowed_classes = {
            0: "person",
            2: "car",
            3: "motorcycle",
            5: "bus",
            1: "bicycle"
        }

    def detect(self, frame, conf_threshold=0.5, min_truck_area=5000):
        # conf_threshold: minimum confidence for detection
        # min_truck_area: minimum area for a truck to be considered valid
        results = self.model(frame, verbose=False)

        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0]) if hasattr(box, 'conf') else 1.0
                if cls in self.allowed_classes and conf >= conf_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    w = x2 - x1
                    h = y2 - y1
                    area = w * h
                    label = self.allowed_classes[cls]
                    # Post-processing: filter unlikely truck detections by area
                    if label == "truck" and area < min_truck_area:
                        continue
                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "class": label,
                        "confidence": conf
                    })
        return detections