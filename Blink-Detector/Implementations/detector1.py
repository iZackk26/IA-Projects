import cv2, dlib, imutils, time
import f_detector
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.video import VideoStream


def main():
    detector = f_detector.eye_blink_detector()
    RECORDED = 0
    COUNTER = 0
    TOTAL = 0
    LIVE = False
    vs = VideoStream(src=0).start()

    while True:
        star_time = time.time()
        im = vs.read()
        im = cv2.flip(im, 1)
        im = imutils.resize(im, width=720)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # detectar_rostro    
        rectangles = detector.detector_faces(gray, 0)
        boxes_face = f_detector.convert_rectangles2array(rectangles,im)
        if len(boxes_face) != 0:
            areas = f_detector.get_areas(boxes_face)
            index = np.argmax(areas)
            rectangles = rectangles[index]
            boxes_face = np.expand_dims(boxes_face[index],axis=0)

            COUNTER, TOTAL = detector.eye_blink(gray, rectangles, COUNTER, TOTAL)

            if TOTAL > 0 and not LIVE:
                LIVE = True
                RECORDED = TOTAL
                img_post = f_detector.bounding_box(im,boxes_face,['Live: '])
            elif TOTAL == RECORDED and ((time.time() - star_time) > 10):
                LIVE = False
                img_post = f_detector.bounding_box(im,boxes_face,['Spoof: '])
        
        else:
            img_post = im 
        # visualizacion 
        cv2.imshow('blink_detection',img_post)
        if cv2.waitKey(1) &0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
