import dlib
import cv2
import numpy as np
from Cam.cam import Camera
from Yunet.yunet import YuNet
from Blinker.blink_detection import EyeDetector

info = ""
counter = 0
total = 0
last_detected = "Hola"


def main():
    global info, counter, total
    model = YuNet(
        modelPath="Yunet/face_detection_yunet_2023mar.onnx",
        inputSize=[320, 320],
        confThreshold=0.9,
        nmsThreshold=0.3,
        topK=100,
        backendId=cv2.dnn.DNN_BACKEND_OPENCV,
        targetId=cv2.dnn.DNN_TARGET_CPU,
    )
    cam = Camera(model, 0, info) #
    faces = []
    for raw_faces, frame, gray in cam.webcam():
        blinker = EyeDetector(raw_faces, frame, gray, counter, total)
        img_post = blinker.detect_blink()
        if img_post is not None and img_post.size != 0:
            cv2.imshow("Blink Detection", img_post)






if __name__ == "__main__":
    main()
