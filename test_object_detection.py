# Test script for verifying YOLOv5 and COCO names integration
import torch
from models.yolo import Model
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
import cv2

# Load YOLO model
model = Model('yolov5s.yaml', nc=80)  # load FP32 model
model.load_state_dict(torch.load('yolov5s.pt')['model'].float().state_dict())

# Load image
img = LoadImages('data/images/sample.jpg', img_size=640)

# Run inference
pred = model(img)[0]
pred = non_max_suppression(pred, 0.25, 0.45)

# Process detections
for i, det in enumerate(pred):
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img.shape).round()

        # Print results
        for *xyxy, conf, cls in reversed(det):
            label = f'{model.names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img, label=label, color=[255,0,0], line_thickness=3)

# Save output image
cv2.imwrite('output.jpg', img)