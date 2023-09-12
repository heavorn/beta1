from models.detect import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
# model = YOLO('path/to/dir/detect/train4/weights/best.pt')

# Validate the model
metrics = model.val(data='data.yaml', device='mps')  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category
