from ultralytics import YOLO


class SimpleTracker:

    def __init__(self, model="yolov8s.pt"):

        self.model = YOLO(model)

        self.class_map = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            25: "umbrella"
        }


    def update(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        tracks = []

        for r in results:
            boxes = r.boxes
            if boxes.id is None:
                continue
            for box, track_id, cls in zip(boxes.xyxy, boxes.id, boxes.cls):
                x1, y1, x2, y2 = box.tolist()
                class_id = int(cls)
                if class_id not in self.class_map:
                    continue
                tracks.append({
                    "id": int(track_id),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "class": self.class_map[class_id]
                })

        # Merge overlapping boxes with the same class (simple IoU-based NMS)
        def iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
            return iou

        merged_tracks = []
        used = set()
        for i, t1 in enumerate(tracks):
            if i in used:
                continue
            merged = t1.copy()
            for j, t2 in enumerate(tracks):
                if i != j and t1["class"] == t2["class"] and iou(t1["bbox"], t2["bbox"]) > 0.5:
                    used.add(j)
            merged_tracks.append(merged)
        return merged_tracks