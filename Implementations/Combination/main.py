import dlib
import cv2
import numpy as np
from Cam.cam import Camera
from Yunet.yunet import YuNet
from Blinker.f_detector import eye_blink_detector

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
    cam = Camera(model, 0, info)  #
    eye = eye_blink_detector()
    faces = []
    for raw_faces, frame, gray in cam.webcam():
        img_post = frame
        processed_faces = []
        if raw_faces is not None:
            for face in raw_faces:
                x, y, w, h = face[
                    :4
                ]  # Tomar las primeras 4 coordenadas que son x, y, w, h
                x1, y1 = int(x), int(y)  # Coordenada superior izquierda
                x2, y2 = int(x + w), int(y + h)  # Coordenada inferior derecha
                processed_face = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
                processed_faces.append(processed_face)
            boxes_face = eye.convert_rectangles2array(processed_faces, frame)
            if len(boxes_face) > 0:
                areas = eye.get_areas(boxes_face)
                boxes_face = np.expand_dims(boxes_face[0], axis=0)
                counter, total = eye.eye_blinker(
                    processed_faces[0], gray, counter, total
                )
                img_post = eye.bounding_box(frame, boxes_face, [last_detected])
                print("Counter: ", counter, "Total: ", total)
                if img_post is not None and img_post.size != 0:
                    cv2.imshow("Blinker", img_post)
        else:
            img_post = frame


if __name__ == "__main__":
    main()
