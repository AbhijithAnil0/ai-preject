import cv2
import json

from pipeline.video_reader import VideoReader
from pipeline.inference_pipeline import InferencePipeline


video_path = r"C:\Users\abhij\Downloads\Telegram Desktop\video_2026-03-21_10-05-06.mp4"

reader = VideoReader(video_path)
pipeline = InferencePipeline()

frame_id = 0

# Changed back to 0 because skipping frames breaks action recognition!
skip_frames = 0 # Change this to 1, 2, or 3 to adjust the speed

while True:

    ret, frame = reader.read()

    if not ret:
        break

    # Process 1 out of every (skip_frames + 1) frames
    if frame_id % (skip_frames + 1) == 0:
        frame = pipeline.process_frame(frame, frame_id)

        cv2.imshow("SEIVR++", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    frame_id += 1

cv2.destroyAllWindows()

events = pipeline.event_engine.event_store.get_all()

print("\n===== STORED EVENTS =====\n")

for e in events:
    print(e)

print(f"\nTotal events: {len(events)}")

# -------- SAVE EVENTS TO JSON --------
with open("events.json", "w") as f:
    json.dump(events, f, indent=4)

print("\n events.json saved successfully")