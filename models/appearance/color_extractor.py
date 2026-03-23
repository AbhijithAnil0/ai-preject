import cv2
import numpy as np

def extract_color(crop):

    if crop is None or crop.size == 0:
        return "unknown"

    # resize for speed
    img = cv2.resize(crop, (80,80))

    # focus on central region (remove road/background)
    h, w = img.shape[:2]
    img = img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    pixels = hsv.reshape((-1,3))
    pixels = np.float32(pixels)

    # k-means clustering to find dominant color
    K = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    _, labels, centers = cv2.kmeans(
        pixels,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    # pick largest cluster
    counts = np.bincount(labels.flatten())
    dominant = centers[np.argmax(counts)]

    h, s, v = dominant

    # neutral colors first (higher saturation tolerance for metallic/wet reflections)
    if s < 85 and v > 120:
        return "white"

    if v < 60:
        return "black"

    if s < 85:
        return "gray"

    # hue-based classification
    if h < 10 or h > 170:
        return "red"

    if 10 <= h < 25:
        return "orange"

    if 25 <= h < 35:
        return "yellow"

    if 35 <= h < 85:
        return "green"

    if 85 <= h < 130:
        return "blue"

    if 130 <= h < 170:
        return "purple"

    return "unknown"