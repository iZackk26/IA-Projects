import dlib
import cv2
import numpy as np
from Cam.cam import Camera
from Yunet.yunet import YuNet
from Blinker.blink_detection import EyeDetector

info = ""
counter = 0
total = 0
last_detected = ""


def bounding_box(img, box, match_name=[]):
    for i in np.arange(len(box)):
        x0, y0, x1, y1 = box[i]
        img = cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 3)
        if not match_name:
            continue
        else:
            box = cv2.putText(
                img,
                match_name[i],
                (x0, y0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            return box

def blink(raw_faces, frame, gray):
    blinker = EyeDetector(raw_faces, frame, gray, counter, total)
    frame, boxes_faces, last_detected = blinker.detect_blink()
    img_post = bounding_box(frame, boxes_faces, last_detected)
    return img_post


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

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if frame is not None:
            h, w = frame.shape
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            model.setInputSize([w, h])
            faces = model.infer(frame)
            # Call the blink function and pass the faces, frame, and gray image
            img_post = blink(faces, frame, gray)
            if img_post is not None and img_post.size != 0:
                frame = img_post

        else:
            img_post = frame



        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()








    cam = Camera(model, 0, info)  #
    for raw_faces, frame, gray in cam.webcam():
        

if __name__ == "__main__":
    main()
