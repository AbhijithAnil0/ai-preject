import cv2
import numpy as np

video_path = "test3.mp4"

cap = cv2.VideoCapture(video_path)

points = []

def mouse_click(event, x, y, flags, param):
    global points, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        print(f"Point added: {x}, {y}")

        cv2.circle(frame, (x, y), 5, (0,255,0), -1)

        if len(points) > 1:
            cv2.line(frame, points[-2], points[-1], (0,255,0), 2)

        cv2.imshow("Video", frame)

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", mouse_click)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Video", frame)

    key = cv2.waitKey(30)

    if key == 32:  # SPACE = pause
        cv2.waitKey(0)

    if key == ord('c'):  # close polygon
        if len(points) > 2:
            cv2.line(frame, points[-1], points[0], (0,255,0), 2)

            print("\nPolygon zone coordinates:")
            print(points)
            print("-------------------------")

            cv2.imshow("Video", frame)

    if key == 27:  # ESC = exit
        break

cap.release()
cv2.destroyAllWindows()