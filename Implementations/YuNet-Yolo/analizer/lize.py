from ultralytics import YOLO
# from ultralytics.engine.results import Results


def analize(img, model_selected):
    if model_selected == "ImageQuality":
        model = YOLO("Models/ImageQuality.pt")
    else:
        model = YOLO("Models/3DFaceModel.pt")

    results = model(img, verbose=False)
    # return True
    return results[-1].probs.top1
