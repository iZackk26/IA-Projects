from eye_blink_detection import detect_blinks
from yolo_models import live_face_detector


for last_detected in detect_blinks():
    print(last_detected)
    if last_detected == "Live":
        break

for last_detected in live_face_detector():
    print(last_detected, type(last_detected))
    if last_detected == "Live":
        break
