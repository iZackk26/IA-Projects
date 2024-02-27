import ultralytics
from ultralytics import YOLO
import subprocess

dataset = 'Live-Spoof/'
command = 'YOLO task=classify mode=train model=yolov8n-cls.pt data={dataset} epochs=50 imgsz=12'


## Entrenar el modelo:
# model = YOLO('yolov8n-cls.pt') 
# model.train(data=dataset, epochs=50, imgsz=12)

## Modelo Generado en la carpeta runs/classify/train2;
# model = YOLO('runs/classify/train2/weights/best.pt')
## Validar el modelo:
# model.val()

# model.export(format='tensorrt')

