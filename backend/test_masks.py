#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from onnx_model_simple import SimpleONNXModel
from PIL import Image
import numpy as np

# Load test image
pil_image = Image.open("person_test.jpg").convert("RGB")
img_array = np.array(pil_image)

# Load model
model = SimpleONNXModel("yolov8n-seg.onnx")

# Test with threshold similar to frontend image
detections = model.predict(img_array, conf_threshold=0.5)

print(f"✅ Found {len(detections)} detections")

for i, det in enumerate(detections):
    print(f"Detection {i+1}:")
    print(f"  Class: {det['class_name']}")
    print(f"  Confidence: {det['confidence']:.3f}")
    print(f"  BBox: {det['bbox']}")
    if det['mask']:
        print(f"  Mask: {len(det['mask'])} points")
    else:
        print(f"  Mask: None")
    print() 