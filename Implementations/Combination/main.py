import cv2
from Cam.cam import Camera
from Yunet.yunet import YuNet

info = ""

def main():
    global info
    model = YuNet(
        modelPath="Yunet/face_detection_yunet_2023mar.onnx",
        inputSize=[320, 320],
        confThreshold=0.9,
        nmsThreshold=0.3,
        topK=100,
        backendId=cv2.dnn.DNN_BACKEND_OPENCV,
        targetId=cv2.dnn.DNN_TARGET_CPU,
    )
    cam = Camera(model, 0, info)
    for faces, frame in cam.webcam():
        print(faces)


if __name__ == "__main__":
    main()
