import cv2
import numpy as np
from analizer.lize import analize
from yunet.yunet import YuNet


def live_face_detector():
    global info
    model = YuNet(
        modelPath="yunet/face_detection_yunet_2023mar.onnx",
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
            h, w, _ = frame.shape
            model.setInputSize([w, h])
            faces = model.infer(frame)
            for det in (
                faces if faces is not None else []
            ):  # No se crashee cuando se sale del frame
                bbox = det[0:4].astype(np.int32)

                # Aument the bounding box
                padding_x = int(0.2 * bbox[2])
                padding_y = int(0.2 * bbox[3])
                x1 = max(bbox[0] - padding_x, 0)
                y1 = max(bbox[1] - padding_y, 0)
                x2 = min(bbox[0] + bbox[2] + padding_x, w)
                y2 = min(bbox[1] + bbox[3] + padding_y, h)
                roi = frame[y1:y2, x1:x2]
                # roi = frame[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
                roi = frame[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
                # cv2.imwrite("roi.jpg", roi)
                if roi.size > 0:
                    if analize(roi) == 0:
                        info = "live"
                    else:
                        info = "spoof"
                # yield info
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(
                    frame,
                    info,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 0),
                    2,
                )

                # cv2.rectangle(
                # frame,
                # (bbox[0], bbox[1]),
                # (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                # (255, 0, 0),
                # 2,
                # )
                # cv2.putText(
                # frame,
                # info,
                # (bbox[0], bbox[1] - 10),
                # cv2.FONT_HERSHEY_SIMPLEX,
                # 0.9,
                # (0, 255, 0),
                # 2,
                # )
            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


live_face_detector()
