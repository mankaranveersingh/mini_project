
# Object Detection Model Comparison on Cityscapes & KITTI

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
import numpy as np
from PIL import Image

# Setup (paths must be configured properly)
CITYSCAPES_PATH = "/path/to/cityscapes"
KITTI_PATH = "/path/to/kitti"

# Define image transform
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# Load sample image for demonstration
sample_image_path = "sample_image.jpg"
sample_image = Image.open(sample_image_path).convert("RGB")
sample_tensor = transform(sample_image).unsqueeze(0)

# Load models (assumes pretrained weights available or models are locally available)
from torchvision.models.detection import fasterrcnn_resnet50_fpn
faster_rcnn = fasterrcnn_resnet50_fpn(pretrained=True).eval()

from yolov5 import YOLOv5  # Ultralytics YOLOv5 (install via pip install yolov5)
yolo5 = YOLOv5("yolov5s.pt")

# Placeholder for other models (e.g., YOLOv3, YOLOv4, YOLOv7)
# These can be implemented using Ultralytics or custom repos

# Detection function
def detect_and_time(model, image_tensor, model_name="Model"):
    start = time.time()
    with torch.no_grad():
        outputs = model(image_tensor)
    end = time.time()
    print(f"{model_name} - Time taken: {end - start:.4f} seconds")
    return outputs

# Detect with Faster R-CNN
faster_rcnn_output = detect_and_time(faster_rcnn, sample_tensor, "Faster R-CNN")

# Detect with YOLOv5
start = time.time()
yolo5_output = yolo5.predict(sample_image_path)
end = time.time()
print(f"YOLOv5 - Time taken: {end - start:.4f} seconds")

# Visualize function
def visualize_fasterrcnn_output(image_tensor, output):
    image = image_tensor.squeeze().permute(1, 2, 0).numpy()
    plt.imshow(image)
    for box in output[0]['boxes']:
        x1, y1, x2, y2 = box
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                          fill=False, edgecolor='red', linewidth=2))
    plt.title("Faster R-CNN Output")
    plt.axis('off')
    plt.show()

visualize_fasterrcnn_output(sample_tensor, faster_rcnn_output)

# Note: For YOLOv3, YOLOv4, YOLOv7 use the respective GitHub repositories:
# - YOLOv3: https://github.com/ultralytics/yolov3
# - YOLOv4: https://github.com/AlexeyAB/darknet
# - YOLOv7: https://github.com/WongKinYiu/yolov7

# Dataset evaluation (optional)
# You can extend this script to run on full datasets (Cityscapes, KITTI)
# and compute mAP using pycocotools or other tools.

# Final output: Log time, mAP, and visualize a few detections for comparison.
print("\n--- Evaluation Complete ---")
