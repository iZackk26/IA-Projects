import ultralytics
from ultralytics import YOLO

# Dataset Path


def train(dataset):
    model = YOLO("yolov8s-cls.pt")
    model.train(data=dataset, epochs=100, imgsz=180, plots=True)


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
