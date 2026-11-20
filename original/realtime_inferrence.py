# run realtime_inferrence.py for qualitatively determining the performance of the trained model or if you want to see the model's performance

from ultralytics import YOLO
import cv2

# load model
model = YOLO('/path/to/model')

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)  # conf threshold = 0.5
    
    annotated_frame = results[0].plot()
    
    cv2.imshow('Sunglasses Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()