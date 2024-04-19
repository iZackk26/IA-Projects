import cv2

class Camera:
    def __init__(self, model, device_number, info):
        self.device_number = device_number
        self.model = model
        self.info = info

    def webcam(self):
        cap = cv2.VideoCapture(self.device_number)
        while True:
            ret, frame = cap.read()
            print(ret)
            if frame is not None:
                h,w,_ = frame.shape
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.model.setInputSize([w, h])
                faces = self.model.infer(frame)
                yield faces, frame, gray

            cv2.imshow(self.info, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
