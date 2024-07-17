import torch
from pathlib import Path
import sys
import numpy as np

# Add the object detection repository to the Python path
sys.path.append('/home/ubuntu/object-detection')
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device

class ObjectDetectionInterface:
    def __init__(self, weights_path='/home/ubuntu/object-detection/best.pt', device=''):
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights_path, device=self.device)
        self.stride = self.model.stride
        self.names = self.model.names
        self.imgsz = check_img_size((640, 640), s=self.stride)  # Adjust image size as needed

    def detect(self, img):
        # Preprocess image
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        pred = self.model(img, augment=False, visualize=False)

        # NMS
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, max_det=1000)

        # Process predictions
        results = []
        for i, det in enumerate(pred):  # per image
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img.shape[2:]).round()
                for *xyxy, conf, cls in reversed(det):
                    results.append({
                        'class': self.names[int(cls)],
                        'confidence': float(conf),
                        'bbox': [float(x) for x in xyxy]
                    })

        return results

def preprocess_image(img):
    # Convert numpy array to the format expected by the model
    # This may need to be adjusted based on the exact format of input images in PikasuBirdAi
    return img.transpose((2, 0, 1))  # HWC to CHW

if __name__ == '__main__':
    # Test the object detection interface

    # Create a dummy image for testing
    test_img = np.random.rand(640, 640, 3).astype(np.float32)

    detector = ObjectDetectionInterface()
    results = detector.detect(preprocess_image(test_img))
    print(results)