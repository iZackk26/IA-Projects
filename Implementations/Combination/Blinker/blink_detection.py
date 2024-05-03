import cv2
import dlib
import numpy as np
from Cam.cam import Camera
from imutils import face_utils
from scipy.spatial import distance as dist
import config as cfg

class EyeDetector:
    _instance = None  # Variable de clase para almacenar la instancia Singleton

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EyeDetector, cls).__new__(cls)
        return cls._instance

    def __init__(self, raw_faces, frame, gray, counter, total):
        # Inicializa la instancia solo una vez.
        if not hasattr(self, 'initialized'):
            self.raw_faces = raw_faces
            self.frame = frame
            self.gray = gray
            self.initialized = True
            self.counter = counter
            self.last_detected = "Hola"
            self.total = total
            self.predict_eyes = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Blink Processing
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
                self.counter += 1
            else:
                if self.counter >= cfg.EYE_AR_CONSEC_FRAMES:
                    self.total += 1
                self.counter = 0

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
    def bounding_box(self, img, box, match_name=[]):
        for i in np.arange(len(box)):
            x0, y0, x1, y1 = box[i]
            img = cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 3)
            if not match_name:
                continue
            else:
                cv2.putText(
                    img,
                    match_name[i],
                    (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
    # Blink Detection
    def process_images(self):
        processed_faces = []
        if self.raw_faces is not None:
            for face in self.raw_faces:
                x,y,w,h = face[:4] # Error
                print("Coordinates",x,y,w,h)
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                processed_face = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
                #print("face",processed_face)
                processed_faces.append(processed_face)
        #print("faces:",processed_faces)
        return processed_faces

    def detect_blink(self):
        img_post = self.frame
        processed_faces = self.process_images()
        print(processed_faces)
        if processed_faces:
            boxes_faces = self.convert_rectangles2array(self.process_images(), self.frame)
            areas = self.get_areas(boxes_faces)
            boxes_faces = np.expand_dims(boxes_faces[0], axis=0)
            self.eye_blinker(self.process_images()[0], self.gray, self.counter, self.total)
            img_post = self.bounding_box(self.frame, boxes_faces, [self.last_detected])
            if img_post is not None and img_post.size != 0:
                print("Guardando imagen")
                cv2.imwrite("test_output.jpg", img_post)
                return img_post
        else:
            img_post = self.frame




