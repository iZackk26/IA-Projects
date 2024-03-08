import ultralytics
from ultralytics import YOLO

# Dataset Path


def train(dataset):
<<<<<<< HEAD
    model = YOLO('yolov8s-cls.pt')
    model.train(data=dataset, epochs=100, imgsz=244, batch=16, weight_decay=0.0005)
=======
    model = YOLO("yolov8s-cls.pt")
    model.train(data=dataset, epochs=100, imgsz=180, plots=True)

>>>>>>> f0c0dfd6605e774b847df9095f5d597182a890c3

# Test Model
def val():
    model = YOLO("runs/classify/train/weights/best.pt")
    model.val()


def main():
    dataset = "dataset/"
    # train(dataset)
    val()


if __name__ == "__main__":
    main()
