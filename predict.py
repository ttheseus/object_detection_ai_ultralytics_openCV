#-------------------------------------------------------------# 
# | [ general visualization code ]  
# |    -> dorothy
#-------------------------------------------------------------#

import os

from ultralytics import YOLO
import cv2

video_path = os.path.join([path to video]) # eg: '/', 'home', 'user', 'Documents', 'ai', 'yolo_testai2-main', 'video', 'VIDEO.mp4'
if not os.path.exists(video_path):
    print(f"Error: Video file does not exist at {video_path}")
else:
    print(f"Video file found at {video_path}")

video_path_out = '{}_out.avi'.format(os.path.splitext(video_path)[0])

cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("Error: Could not open video at {video_path}.")
    exit()


ret, frame = cap.read()
if not ret or frame is None:
    print("Error: Could not read the first frame.")
    exit()

H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MJPG'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join([path to weights]) # eg. '/', 'home', 'user', 'runs', 'detect', 'train19', 'weights', 'last.pt'

model = YOLO(model_path)

threshold = 0.3

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 3, cv2.LINE_AA)
    
    out.write(frame)
    ret, frame = cap.read()

    if ret and frame is not None:
        H, W, _ = frame.shape
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
