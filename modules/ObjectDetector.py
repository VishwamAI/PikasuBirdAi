from .object_detection_interface import ObjectDetectionInterface, preprocess_image

class ObjectDetector:
    def __init__(self):
        self.detector = ObjectDetectionInterface()

    def detect_objects(self, image):
        preprocessed_image = preprocess_image(image)
        return self.detector.detect(preprocessed_image)