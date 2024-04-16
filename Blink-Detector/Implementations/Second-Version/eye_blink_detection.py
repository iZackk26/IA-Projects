from imutils.video import VideoStream
import cv2
import time
import f_detector
import imutils
import numpy as np

# instancio detector
detector = f_detector.eye_blink_detector()
# iniciar variables para el detector de parapadeo
COUNTER = 0
TOTAL = 0

# ----------------------------- video -----------------------------
# ingestar data
vs = VideoStream(src=0).start()
last_blink = 0
last_detected = ""
spoof_interval = 5
start_time = time.time()

while True:
    im = vs.read()
    im = cv2.flip(im, 1)
    im = imutils.resize(im, width=720)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # detectar_rostro
    rectangles = detector.detector_faces(gray)
    boxes_face = f_detector.convert_rectangles2array(rectangles, im)
    if len(boxes_face) != 0:
        # seleccionar el rostro con mas area
        areas = f_detector.get_areas(boxes_face)
        index = np.argmax(areas)
        rectangles = rectangles[index]
        boxes_face = np.expand_dims(boxes_face[index], axis=0)
        # blinks_detector
        COUNTER, TOTAL = detector.eye_blink(gray, rectangles, COUNTER, TOTAL)
        # agregar bounding box
        img_post = f_detector.bounding_box(im, boxes_face, [last_detected])
        # print(time.time(), start_time)

        if last_blink < TOTAL:
            last_blink = TOTAL
            last_detected = "Live"
        elif time.time() - start_time >= spoof_interval:
            start_time = time.time()
            last_detected = "Spoof"

    else:
        img_post = im
    # visualizacion
    cv2.imshow("blink_detection", img_post)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
