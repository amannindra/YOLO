import cv2
import math 

 
from ultralytics import YOLO

model = YOLO('yolov8s.pt')

cam = cv2.VideoCapture(0)


cam.set(4, 1920)

#yolo TASK MODE ARGS

if not cam.isOpened():
    print("Cannot open Camera")
    exit()


while True:
    success, frame = cam.read() 

    if not success:
        continue
    results = model.track(frame, stream = True)

    for result in results:
        classes_names = result.names
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            confidence = math.ceil(box.conf[0] * 100)/100
            print("Confidence -->", confidence)

            cls = int(box.cls[0])
            print("Class name -->", classes_names[cls])

            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(frame, classes_names[cls], org, font, fontScale, color, thickness)


    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()

print(cam)
