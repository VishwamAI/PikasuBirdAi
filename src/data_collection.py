# Note: Make sure to install pycocotools using: pip install pycocotools
import os
import json
from PIL import Image
import torch
from torchvision import transforms
from pycocotools.coco import COCO
from transformers import DetrForObjectDetection, ViTImageProcessor, YolosImageProcessor
from ultralytics import YOLO
import subprocess

# Define paths for data storage and COCO dataset
image_dir = 'data/images'
label_dir = 'data/labels'
coco_annotation_file = 'annotations/instances_val2017.json'
coco_images_dir = 'val2017'

# Define the models to use for data collection
detr_model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
yolos_processor = YolosImageProcessor.from_pretrained('hustvl/yolos-small')

# Function to collect and preprocess data
def collect_data():
    # Check if the annotation file and images directory exist
    if not os.path.exists(coco_annotation_file):
        raise FileNotFoundError(f"COCO annotation file not found: {coco_annotation_file}")
    if not os.path.exists(coco_images_dir):
        raise FileNotFoundError(f"COCO images directory not found: {coco_images_dir}")

    # Load COCO dataset from local files
    coco = COCO(coco_annotation_file)

    # Define preprocessing steps
    preprocess = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Collect and preprocess data
    processed_count = 0
    target_count = 1000
    for idx, img_id in enumerate(coco.getImgIds()):
        try:
            # Process COCO dataset item
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(coco_images_dir, img_info['file_name'])
            coco_image = Image.open(img_path).convert('RGB')
            coco_preprocessed = preprocess(coco_image)
            print(f"Image tensor shape: {coco_preprocessed.shape}")

            # Save preprocessed image
            save_image(coco_preprocessed, f"{image_dir}/train/coco_{idx}.jpg")

            # Get annotations for the image
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            # Save annotations
            save_annotation(anns, f"{label_dir}/train/coco_{idx}.txt")

            processed_count += 1
            if processed_count >= target_count:
                break
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")

    if processed_count == target_count:
        print(f"Successfully collected and preprocessed {processed_count} images from COCO dataset")
    else:
        print(f"Warning: Only processed {processed_count} out of {target_count} target images")

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

# Function to create COCO bird names file
def create_coco_bird_names():
    bird_classes = {
        14: 'bird',
        15: 'cat',  # Including cat for reference
        16: 'dog',  # Including dog for reference
        17: 'horse',  # Including horse for reference
        18: 'sheep',  # Including sheep for reference
        19: 'cow',  # Including cow for reference
    }
    with open('coco_bird_names.txt', 'w') as f:
        for class_id, name in bird_classes.items():
            f.write(f"{class_id}:{name}\n")

# Function to use COCO dataset names for object labeling
def use_coco_names():
    if not os.path.exists('coco_bird_names.txt'):
        create_coco_bird_names()

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
            try:
                class_id, *bbox = line.strip().split()
                class_name = coco_names.get(int(class_id) if class_id.isdigit() else class_id, 'unknown')
                f.write(f"{class_name} {' '.join(bbox)}\n")
            except ValueError as e:
                print(f"Error processing class_id: {class_id} in file {filename}. Error: {e}")
                f.write(f"unknown {' '.join(bbox)}\n")

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

def verify_collected_data():
    collected_images = os.listdir(os.path.join(image_dir, 'train'))
    print(f"Collected {len(collected_images)} images")

    collected_labels = os.listdir(os.path.join(label_dir, 'train'))
    print(f"Collected {len(collected_labels)} label files")

    for label_file in collected_labels:
        if os.path.isfile(os.path.join(label_dir, 'train', label_file)):
            sample_label = os.path.join(label_dir, 'train', label_file)
            with open(sample_label, 'r') as f:
                print(f"Sample label content from {label_file}:\n{f.read()}")
            break
    else:
        print("No valid label files found.")

def grep_collected_data(pattern):
    grep_command = f"grep -r '{pattern}' {label_dir}"
    result = subprocess.run(grep_command, shell=True, capture_output=True, text=True)
    print(f"Grep results for pattern '{pattern}':\n{result.stdout}")

def verify_coco_format():
    # Check if the collected data matches the expected COCO format
    coco_format_valid = True
    for label_file in os.listdir(os.path.join(label_dir, 'train')):
        if label_file.endswith('.txt'):
            with open(os.path.join(label_dir, 'train', label_file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5 or not all(part.replace('.', '', 1).isdigit() for part in parts):
                        print(f"Invalid format in {label_file}: {line.strip()}")
                        coco_format_valid = False
                        break

    if coco_format_valid:
        print("All collected data matches the expected COCO format.")
    else:
        print("Some collected data does not match the expected COCO format.")

# Main execution
if __name__ == '__main__':
    # Ensure necessary directories exist
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    if os.path.exists(coco_annotation_file) and os.path.isdir(coco_images_dir):
        collect_data()
        integrate_yolo()
        use_coco_names()
        custom_dataset = create_custom_dataset_class()
        print(f"Created custom dataset with {len(custom_dataset)} samples")
        verify_collected_data()
        grep_collected_data("bird")  # Example pattern
    else:
        print(f"Error: COCO dataset files not found. Please ensure the annotation file '{coco_annotation_file}' and images directory '{coco_images_dir}' exist.")