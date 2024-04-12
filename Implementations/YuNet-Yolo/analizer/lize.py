from ultralytics import YOLO
# from ultralytics.engine.results import Results

model = YOLO("Models/ImageQuality.pt")


def analize(img):
    results = model(img)
    print(results[-1].probs.top1)
    # return True
    return results[-1].probs.top1
