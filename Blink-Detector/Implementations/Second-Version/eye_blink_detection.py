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
cv2.setUseOptimized(True)
cv2.setNumThreads(4)
cv2.ocl.setUseOpenCL(True)
# ----------------------------- video -----------------------------
# ingestar data
vs = VideoStream(src=0).start()
star_time = time.time()
img_post = None
while True:
    im = vs.read()
    im = cv2.flip(im, 1)
    im = imutils.resize(im, width=720)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # detectar_rostro
    rectangles = detector.detector_faces(gray, 0)
    boxes_face = f_detector.convert_rectangles2array(rectangles, im)
    if len(boxes_face) != 0:
        # seleccionar el rostro con mas area
        areas = f_detector.get_areas(boxes_face)
        index = np.argmax(areas)
        rectangles = rectangles[index]
        boxes_face = np.expand_dims(boxes_face[index], axis=0)
        # blinks_detector
        blink_detected = detector.eye_blink(gray, rectangles, COUNTER, TOTAL)
        print(detector.eye_blink(gray, rectangles, COUNTER, TOTAL))

        if blink_detected:
            img_post = f_detector.bounding_box(im, boxes_face, ["Live"])
            star_time = time.time()
        elif time.time() - star_time >= 10:
            img_post = f_detector.bounding_box(im, boxes_face, ["Spoof"])
            star_time = time.time()
    else:
        img_post = im
    if img_post is not None:
        cv2.imshow("blink_detection", img_post)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
