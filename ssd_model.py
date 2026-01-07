import tensorflow as tf
import numpy as np

class SSDModel:
    def __init__(self, model_path):
        # Load the pre-trained SSD model
        self.model = tf.saved_model.load(model_path)

    def detect_objects(self, image):
        # Convert image to tensor
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Perform detection
        detections = self.model(input_tensor)

        # Extract detection results
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.int32)

        return boxes, scores, classes

# Example usage
# model = SSDModel('path/to/ssd_mobilenet_v2/saved_model')
# boxes, scores, classes = model.detect_objects(image)
