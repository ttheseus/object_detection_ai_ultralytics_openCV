#-------------------------------------------------------------# 

# | [ minecraft mob detection AI - training code]  
# |    -> dorothy

# | useful video follow along for the code that i used: 
# | https://www.youtube.com/watch?v=Z-65nqxUdl4&t=1979s
# |
# | go into terminal and enter 'pip install ultralytics'

#-------------------------------------------------------------#
from ultralytics import YOLO 

# Load a model
model = YOLO("yolov8n.pt") # build model from scratch

results = model.train(data="config.yaml", epochs=3) #train the model
