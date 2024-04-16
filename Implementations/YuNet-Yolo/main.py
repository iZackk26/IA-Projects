from yolo_models import live_face_detector
import time


veredict = []


def yolo_classifiers(model_selected):
    start = time.time()
    image_quality_outputs = []
    for last_detected in live_face_detector(model_selected):
        image_quality_outputs.append(last_detected)
        if time.time() - start > 10:
            break

    if image_quality_outputs.count("Live") > image_quality_outputs.count("Spoof"):
        print(f"{model_selected} model detected you as Live")
        veredict.append("Live")
    else:
        print("Image Quality model detected you as Spoof")
        veredict.append("Spoof")

    print("---------")


yolo_classifiers("ImageQuality")
yolo_classifiers("3DFaces")

if veredict.count("Live") > veredict.count("Spoof"):
    print("You are Live")
else:
    print("You are Spoof")
