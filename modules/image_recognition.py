import numpy as np
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from PIL import Image

class ImageRecognizer:
    def __init__(self):
        self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    def process_image(self, observation):
        # Ensure the image is in the correct format (e.g., PIL Image)
        image = Image.fromarray(observation.astype('uint8'), 'RGB')

        # Preprocess the image
        inputs = self.processor(images=image, return_tensors="pt")

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get the predicted class
        logits = outputs.logits
        predicted_class = logits.argmax(-1).item()

        return predicted_class