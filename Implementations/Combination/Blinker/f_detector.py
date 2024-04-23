import numpy as np
import dlib
import config as cfg
from imutils import face_utils
from scipy.spatial import distance as dist

class eye_blink_detector:
    def __init__(self):
        self.predict_eyes = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def eye_blinker(self, rect, gray, counter, total):
        print(type(rect))
        shape = self.predict_eyes(gray, rect)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        left_eye_ratio = self.eye_aspect_ratio(left_eye)
        right_eye_ratio = self.eye_aspect_ratio(right_eye)

        eye_ratio = (left_eye_ratio + right_eye_ratio) / 2.0

        if eye_ratio < cfg.EYE_AR_THRESH:
            counter += 1
        else:
            if counter >= cfg.EYE_AR_CONSEC_FRAMES:
                total += 1
            counter = 0
        return counter, total

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def convert_rectangles2array(self, rectangles, image):
        res = np.array([])
        for box in rectangles:
            [x0, y0, x1, y1] = (
                max(0, box.left()),
                max(0, box.top()),
                min(box.right(), image.shape[1]),
                min(box.bottom(), image.shape[0]),
            )
            new_box = np.array([x0, y0, x1, y1])
            if res.size == 0:
                res = np.expand_dims(new_box, axis=0)
            else:
                res = np.vstack((res, new_box))
        return res
    
    def get_areas(self, boxes):
        areas = []
        for box in boxes:
            x0,y0,x1,y1 = box
            area = (y1 - y0) * (x1 - x0)
            areas.append(area)
        return areas
