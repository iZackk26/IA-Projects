from ultralytics import YOLO
# from ultralytics.engine.results import Results


def get_probs(img):
    model = YOLO("ImageQuality.pt")
    results = model(img, verbose=False)
    # return True
    return results[-1].probs.top1
