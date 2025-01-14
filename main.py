#-------------------------------------------------------------# 
# | [ minecraft mob detection AI - training code]  
# |    -> dorothy
#-------------------------------------------------------------#

from ultralytics import YOLO 

# Load a model
model = YOLO("yolov8n.pt") # build model from scratch

results = model.train(data="config.yaml", epochs=3) #train the model
