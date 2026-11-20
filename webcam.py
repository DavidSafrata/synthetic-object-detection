import sys
import cv2
from ultralytics import YOLO

model = YOLO(sys.argv[1])
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press 'q' to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, conf=0.5, verbose=False)
    cv2.imshow("Sunglasses Detection", results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
