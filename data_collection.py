import os
import json
import requests
from PIL import Image
import torch
from torchvision import transforms
from datasets import load_dataset
from transformers import DetrForObjectDetection, ViTFeatureExtractor, YolosFeatureExtractor
from ultralytics import YOLO

# Define paths for data storage
image_dir = 'data/images'
label_dir = 'data/labels'

# Define the models to use for data collection
detr_model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
vit_model = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
yolos_model = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')

# Function to collect and preprocess data
def collect_data():
    # Load public datasets
    coco_dataset = load_dataset("coco", split="train")
    inaturalist_dataset = load_dataset("inaturalist", "2021_train", split="train")

    # Define preprocessing steps
    preprocess = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Collect and preprocess data
    for idx, (coco_item, inat_item) in enumerate(zip(coco_dataset, inaturalist_dataset)):
        try:
            # Process COCO dataset item
            coco_image = Image.open(requests.get(coco_item['image'], stream=True).raw)
            coco_preprocessed = preprocess(coco_image)

            # Process iNaturalist dataset item
            inat_image = Image.open(requests.get(inat_item['image'], stream=True).raw)
            inat_preprocessed = preprocess(inat_image)

            # Save preprocessed images
            save_image(coco_preprocessed, f"{image_dir}/train/coco_{idx}.jpg")
            save_image(inat_preprocessed, f"{image_dir}/train/inat_{idx}.jpg")

            # Save annotations
            save_annotation(coco_item['objects'], f"{label_dir}/train/coco_{idx}.txt")
            save_annotation(inat_item['annotations'], f"{label_dir}/train/inat_{idx}.txt")

            if idx >= 1000:  # Limit to 1000 images for now
                break
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")

def save_image(tensor, filename):
    image = transforms.ToPILImage()(tensor)
    image.save(filename, format='JPEG')

def save_annotation(annotations, filename):
    with open(filename, 'w') as f:
        for ann in annotations:
            f.write(f"{ann['category_id']} {ann['bbox'][0]} {ann['bbox'][1]} {ann['bbox'][2]} {ann['bbox'][3]}\n")

# Function to integrate YOLO model for object detection
def integrate_yolo():
    yolo_model = YOLO('yolov8n.pt')  # Load a pretrained YOLOv8n model

    # Perform object detection on collected images
    for image_file in os.listdir(f"{image_dir}/train"):
        if image_file.endswith('.jpg'):
            results = yolo_model(f"{image_dir}/train/{image_file}")

            # Save YOLO detection results
            save_yolo_results(results, f"{label_dir}/train/{image_file[:-4]}_yolo.txt")

def save_yolo_results(results, filename):
    with open(filename, 'w') as f:
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                x, y, w, h = box.xywh[0]
                f.write(f"{class_id} {x} {y} {w} {h}\n")

# Function to use COCO dataset names for object labeling
def use_coco_names():
    coco_names = {}
    with open('coco_bird_names.txt', 'r') as f:
        for line in f:
            class_id, name = line.strip().split(':')
            coco_names[int(class_id)] = name

    # Update annotation files with COCO names
    for label_file in os.listdir(f"{label_dir}/train"):
        if label_file.endswith('.txt'):
            update_labels_with_names(f"{label_dir}/train/{label_file}", coco_names)

def update_labels_with_names(filename, coco_names):
    with open(filename, 'r') as f:
        lines = f.readlines()

    with open(filename, 'w') as f:
        for line in lines:
            class_id, *bbox = line.strip().split()
            class_name = coco_names.get(int(class_id), 'unknown')
            f.write(f"{class_name} {' '.join(bbox)}\n")

# Function to create a custom dataset class
class CustomBirdDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.image_files[idx][:-4] + '.txt')

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Load and process labels
        with open(label_path, 'r') as f:
            labels = [line.strip().split() for line in f]

        # Convert labels to tensor
        labels = torch.tensor([[float(x) for x in label[1:]] for label in labels])

        return image, labels

def create_custom_dataset_class():
    return CustomBirdDataset(f"{image_dir}/train", f"{label_dir}/train", transform=transforms.ToTensor())

# Main execution
if __name__ == '__main__':
    collect_data()
    integrate_yolo()
    use_coco_names()
    custom_dataset = create_custom_dataset_class()
    print(f"Created custom dataset with {len(custom_dataset)} samples")