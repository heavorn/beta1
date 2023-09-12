from models.detect import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data='data.yaml', epochs=2, imgsz=640, device='mps', fraction=0.1)
