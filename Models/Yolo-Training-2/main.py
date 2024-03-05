import ultralytics
from ultralytics import YOLO

# Dataset Path

def train(dataset):
    model = YOLO('yolov8n-cls.pt')
    model.train(data=dataset, epochs=50, imgsz=12)

# Test Model
def val():
    model = YOLO('runs/classify/train/weights/best.pt')
    model.val()

def main():
    dataset = 'dataset/'
    #train(dataset)
    val()

if __name__ == "__main__":
    main()
