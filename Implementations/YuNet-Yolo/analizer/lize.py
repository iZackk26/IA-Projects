from ultralytics import YOLO
# from ultralytics.engine.results import Results

model = YOLO("/home/izack/Investigations/IA/Implementations/YuNet-Yolo/best.pt")

def analize(img):
    results = model(img)
    print(results[-1].probs.top1conf.item())
    # return True
    return True if results[-1].probs.top1conf > 0.9 else False
