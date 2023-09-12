from models.detect import YOLO

# Load a model
model = YOLO('path/to/dir/detect/train4/weights/best.pt')  # load a best weight
# model = YOLO('yolov8n.pt')  # load a pretrained model

# Train the model with 2 GPUs
results = model.predict(save=True, device='mps', source='old/img/2.jpg')

