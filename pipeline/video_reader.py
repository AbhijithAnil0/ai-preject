import cv2

class VideoReader:

    def __init__(self, path):

        self.cap = cv2.VideoCapture(path)

    def read(self):

        ret, frame = self.cap.read()

        return ret, frame