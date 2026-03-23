import os
import cv2
import torch
import numpy as np
from torchvision import transforms
import yaml

from events.temporal_memory import PersonMemory
from models.tracking.bytetrack_tracker import SimpleTracker
from models.action.vit_action_model import ActionViT
from models.appearance.color_extractor import extract_color
from events.event_engine import EventEngine


# ---------- LOAD ZONES ----------
with open("configs/zones.yaml", "r") as f:
    zone_config = yaml.safe_load(f)

zones = {}

for name, data in zone_config["zones"].items():
    zones[name] = np.array(data["points"])


# ---------- CONSTANTS ----------
VEHICLE_CLASSES = {"car","truck","bus","motorcycle","bicycle"}


# ---------- VIT TRANSFORM ----------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class InferencePipeline:

    def __init__(self):

        self.tracker = SimpleTracker()

        self.prev_zone = {}
        self.entry_side = {}
        self.prev_position = {}
        self.speed_history = {}
        # --------- MULTI-FRAME BUFFER ---------
        self.frame_buffer = {}
        self.buffer_size = 8   # number of frames per person
        self.classes = ['run', 'stand', 'walk']

        # Adaptive speed thresholds
        self.walk_speeds = []
        self.run_speeds = []
        
        # Real-world per-frame movement is extremely small as a fraction of body size at 30fps!
        # A person walking towards/away from the camera has almost 0 center translation!
        self.walk_threshold = 0.001  # 0.1% pixel translation per frame is walking
        self.run_threshold = 0.015   # 1.5% pixel translation per frame is running
        
        self.threshold_update_frame = 0

        checkpoint_path = "runs/vit_action_finetuned.pth"
        if os.path.exists(checkpoint_path):
            self.action_model = ActionViT(checkpoint=checkpoint_path, pretrained=False)
        else:
            self.action_model = ActionViT(pretrained=True)
        self.event_engine = EventEngine()
        self.memory = PersonMemory()

        self.last_actions = {}
        self.action_history = {}  # For temporal smoothing
        self.last_colors = {}

        # Motion state = stabilized run/walk/stand decision per person
        self.motion_state = {}
        self.motion_state_count = {}

        # Hold action counters to prevent brief standing spikes
        self.action_hold = {}

        self.seen_ids = set()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.action_model.to(self.device)


    def update_adaptive_thresholds(self):
        if len(self.walk_speeds) > 10 and len(self.run_speeds) > 10:
            # Use percentiles for adaptive thresholds
            import numpy as np
            walk_sorted = sorted(self.walk_speeds)
            run_sorted = sorted(self.run_speeds)
            self.walk_threshold = np.percentile(walk_sorted, 75)  # 75th percentile of walk speeds
            self.run_threshold = np.percentile(run_sorted, 25)   # 25th percentile of run speeds
            print(f"Updated thresholds: walk={self.walk_threshold:.2f}, run={self.run_threshold:.2f}")
            # Keep only recent samples
            self.walk_speeds = self.walk_speeds[-50:]
            self.run_speeds = self.run_speeds[-50:]


    def process_frame(self, frame, frame_id):

        tracks = self.tracker.update(frame)

        if len(tracks) == 0:
            return frame


        # ---------- FIRST PASS: COLLECT UMBRELLAS ----------
        umbrella_boxes = []

        for t in tracks:
            if t["class"] == "umbrella":
                umbrella_boxes.append(t["bbox"])


        # ---------- SECOND PASS: PROCESS OBJECTS ----------
        for t in tracks:

            if t["class"] == "umbrella":
                continue

            pid = t["id"]
            obj_class = t["class"]

            bbox = t["bbox"]

            # Fix if bbox is tensor
            if isinstance(bbox, torch.Tensor):
                bbox = bbox.cpu().numpy()

            bbox = list(bbox)

            # Safety check
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # ---------- PERSON MOVEMENT SPEED ----------
            prev_xy = self.prev_position.get(pid)
            movement_speed = 0.0
            if prev_xy is not None:
                dx = center_x - prev_xy[0]
                dy = center_y - prev_xy[1]
                movement_speed = (dx*dx + dy*dy) ** 0.5

            self.prev_position[pid] = (center_x, center_y)

            # track speed history to reduce jitter between walk/run
            if pid not in self.speed_history:
                self.speed_history[pid] = []
            self.speed_history[pid].append(movement_speed)
            if len(self.speed_history[pid]) > 8:
                self.speed_history[pid].pop(0)
            avg_speed = sum(self.speed_history[pid]) / len(self.speed_history[pid])

            frame_width = frame.shape[1]


            # ---------- SAFE BOUNDARIES ----------
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)


            # ---------- UMBRELLA OVERLAP ----------
            if obj_class == "person":

                for ub in umbrella_boxes:

                    ux1, uy1, ux2, uy2 = ub

                    # Umbrellas are often held above the person, so we expand the person's
                    # bounding box upward by 50 pixels and sideways by 20 pixels to ensure overlaps touch!
                    margin_x = 20
                    margin_y = 50
                    px1 = x1 - margin_x
                    px2 = x2 + margin_x
                    py1 = y1 - margin_y
                    py2 = y2 + margin_y

                    overlap = not (
                        px2 < ux1 or px1 > ux2 or
                        py2 < uy1 or py1 > uy2
                    )

                    if overlap:
                        obj_class = "person_U"
                        break


            # ---------- ZONE DETECTION (VEHICLES ONLY) ----------
            zone = "none"

            if obj_class in VEHICLE_CLASSES:

                for name, polygon in zones.items():

                    if cv2.pointPolygonTest(polygon, (center_x, center_y), False) >= 0:
                        zone = name
                        break


            # ---------- ENTRY SIDE DETECTION ----------
            if obj_class in VEHICLE_CLASSES and pid not in self.entry_side:

                if center_x < frame_width * 0.25:
                    self.entry_side[pid] = "LEFT_ENTRY"

                elif center_x > frame_width * 0.75:
                    self.entry_side[pid] = "RIGHT_ENTRY"

                else:
                    self.entry_side[pid] = "CENTER_ENTRY"


            # ---------- VEHICLE ROUTE EVENTS ----------
            route_event = None

            if obj_class in VEHICLE_CLASSES:

                previous = self.prev_zone.get(pid)

                if zone != "none" and previous != zone:

                    entry = self.entry_side.get(pid, "UNKNOWN")

                    # Record reaching a path zone
                    if previous is None:
                        route_event = f"VEHICLE_FROM_{entry}_ENTERED_{zone.upper()}"

                    # Record turning path transition
                    elif zone == "left_path":
                        route_event = f"VEHICLE_FROM_{entry}_TURNED_LEFT"

                    elif zone == "right_path":
                        route_event = f"VEHICLE_FROM_{entry}_TURNED_RIGHT"

                    elif zone == "straight_path":
                        route_event = f"VEHICLE_FROM_{entry}_WENT_STRAIGHT"

                    if route_event is not None:
                        print(f"EVENT: {route_event} | ID {pid}")

                self.prev_zone[pid] = zone


            # ---------- CROPPING ----------
            if obj_class == "person" or obj_class == "person_U":

                crop = frame[y1:y2, x1:x2]
                # --------- FRAME BUFFER UPDATE ---------
                if obj_class == "person" or obj_class == "person_U":
                    if pid not in self.frame_buffer:
                        self.frame_buffer[pid] = []

                    self.frame_buffer[pid].append(crop)

                    if len(self.frame_buffer[pid]) > self.buffer_size:
                        self.frame_buffer[pid].pop(0)

            else:

                crop = frame[y1:y2, x1:x2]

            # (Removed local frame writing to disk to massively speed up performance and save space!)

            if crop.size == 0:
                continue


            # ---------- ACTION RECOGNITION USING VIT ----------
            if obj_class == "person" or obj_class == "person_U":
                
                # Fetch all recent frames for this person (up to 8 frames)
                crops = self.frame_buffer.get(pid, [crop])
                
                # Transform each crop and stack into a batch
                t_list = [transform(c) for c in crops]
                t = torch.stack(t_list).to(self.device)
                
                # Predict across the batch (returns average probability across time)
                # By dropping conf_thresh to 0.0, we force the ViT to just pick the highest probability
                # instead of returning "unknown" and defaulting to "stand" every time the confidence is < 0.6
                raw_action = self.action_model.predict(t, conf_thresh=0.0, use_smoothing=False)
                action = raw_action
                
                # Treat low confidence as the previous stable action to avoid flicker
                if action == "unknown":
                    action = self.last_actions.get(pid, "walk")

                # Maintain a history of last 7 actions for each person to smooth the ViT output
                N = 7  
                if pid not in self.action_history:
                    self.action_history[pid] = []
                
                self.action_history[pid].append(action)
                if len(self.action_history[pid]) > N:
                    self.action_history[pid].pop(0)

                # Use a majority vote for stable temporal output
                from collections import Counter
                top_action, top_count = Counter(self.action_history[pid]).most_common(1)[0]
                
                if top_count >= 4:
                    smoothed_action = top_action
                else:
                    smoothed_action = self.last_actions.get(pid, top_action)

                self.last_actions[pid] = smoothed_action
                action = smoothed_action
                
                self.motion_state[pid] = action

                # Action hold buffer: require 6 frames for changes away from walk/run
                hold_state = self.action_hold.get(pid, {"action": action, "count": 1})
                if hold_state["action"] == action:
                    hold_state["count"] += 1
                else:
                    hold_state = {"action": action, "count": 1}

                if self.last_actions.get(pid) in ["walk", "run"] and action == "stand" and hold_state["count"] < 6:
                    action = self.last_actions[pid]

                if self.last_actions.get(pid) == "run" and action == "walk" and hold_state["count"] < 5:
                    action = "run"

                if self.last_actions.get(pid) == "walk" and action == "run" and hold_state["count"] < 4:
                    action = "walk"

                hold_state["action"] = action
                self.action_hold[pid] = hold_state
                self.last_actions[pid] = action

                # We no longer need to collect and update speed thresholds 
                # because the Vision Transformer handles action classification directly!
            else:
                action = "vehicle"


            # ---------- VEHICLE COLOR ----------
            if obj_class in VEHICLE_CLASSES:

                if pid not in self.last_colors:

                    color = extract_color(crop)
                    self.last_colors[pid] = color

                else:

                    color = self.last_colors.get(pid, "unknown")

            else:

                color = None


            # ---------- TEMPORAL MEMORY ----------
            self.memory.update(pid, action, frame_id)


            # ---------- EVENT ENGINE ----------
            self.event_engine.process(
                pid,
                obj_class,
                action,
                color,
                [x1, y1, x2, y2],
                frame_id,
                route_event=route_event
            )


            # ---------- VISUALIZATION ----------
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            if obj_class == "person" or obj_class == "person_U":
                label = f"{obj_class} ID {pid} {action}"

            else:
                label = f"{obj_class} ID {pid} {color} {zone}"

            cv2.putText(
                frame,
                label,
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )


        return frame