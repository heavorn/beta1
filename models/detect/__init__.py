# yolov8 YOLO 🚀, AGPL-3.0 license

from engine.model import Model
from nn.tasks import DetectionModel
from .predict import DetectionPredictor
from .train import DetectionTrainer
from .val import DetectionValidator


class YOLO(Model):
    """
    YOLO (You Only Look Once) object detection model.
    """

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes"""
        return {
                'detect': {
                'model': DetectionModel,
                'trainer': DetectionTrainer,
                'validator': DetectionValidator,
                'predictor': DetectionPredictor, },
            
            	}
