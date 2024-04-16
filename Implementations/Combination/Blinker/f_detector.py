import numpy as np
from imutils import face_utils
import dlib
from scipy.spatial import distance as dist

class eye_blink_detector:
    def __init__(self):
        self.predict_eyes = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def rectangles(self,faces):
        if faces is not None:
            x_center, y_center, width, height = faces[0][:4]
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            rec_box = np.array([x1, y1, x2, y2])
        else:
            return None
        return rec_box
