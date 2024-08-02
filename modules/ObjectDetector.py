from .object_detection_interface import ObjectDetectionInterface, preprocess_image
import numpy as np
from typing import List, Union

class ObjectDetector:
    def __init__(self):
        self.detector = ObjectDetectionInterface()

    def detect_objects(self, images: Union[np.ndarray, List[np.ndarray]]):
        try:
            if isinstance(images, np.ndarray):
                images = [images]

            preprocessed_images = [preprocess_image(img) for img in images]
            batch_results = self.detector.detect_batch(preprocessed_images)

            return [self._post_process(result) for result in batch_results]
        except Exception as e:
            print(f"Error in object detection: {str(e)}")
            return []

    def _post_process(self, detection_result):
        # Implement post-processing logic here (e.g., non-max suppression, filtering)
        return detection_result
