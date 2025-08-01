#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image

# Load test image
pil_image = Image.open("person_test.jpg").convert("RGB")
img_array = np.array(pil_image)
print(f"✅ Image loaded: {img_array.shape}")

# Load ONNX session
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession("yolov8n-seg.onnx", session_options, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# Preprocess
img_resized = cv2.resize(img_array, (640, 640), interpolation=cv2.INTER_LINEAR)
img_norm = img_resized.astype(np.float32) / 255.0
img_batch = np.transpose(img_norm, (2, 0, 1))[None, ...]

# Run inference
outputs = session.run(None, {input_name: img_batch})

# Check raw outputs
print(f"📊 Number of outputs: {len(outputs)}")
print(f"📊 Output 0 shape: {outputs[0].shape}")
if len(outputs) > 1:
    print(f"📊 Output 1 shape: {outputs[1].shape}")

# Parse detections
detections = outputs[0][0].T  # Shape: (8400, 116)
boxes = detections[:, :4]  # xywh
class_scores = detections[:, 4:84]  # 80 classes

print(f"📊 Detections shape: {detections.shape}")
print(f"📊 Boxes shape: {boxes.shape}")
print(f"📊 Class scores shape: {class_scores.shape}")

# Check confidence distribution
max_scores = np.max(class_scores, axis=1)
print(f"📊 Max confidence in raw scores: {np.max(max_scores):.4f}")
print(f"📊 Min confidence in raw scores: {np.min(max_scores):.4f}")
print(f"📊 Mean confidence in raw scores: {np.mean(max_scores):.4f}")

# Apply sigmoid to see actual confidences
confidences = 1 / (1 + np.exp(-max_scores))
print(f"📊 Max confidence after sigmoid: {np.max(confidences):.4f}")
print(f"📊 Min confidence after sigmoid: {np.min(confidences):.4f}")
print(f"📊 Mean confidence after sigmoid: {np.mean(confidences):.4f}")

# Check how many pass different thresholds
for threshold in [0.01, 0.1, 0.25, 0.5]:
    count = np.sum(confidences > threshold)
    print(f"📊 Detections above {threshold}: {count}")

# Show top 10 detections
top_indices = np.argsort(confidences)[::-1][:10]
print(f"\n🔝 Top 10 detections:")
for i, idx in enumerate(top_indices):
    class_idx = np.argmax(class_scores[idx])
    print(f"  {i+1}. Class {class_idx}, Confidence: {confidences[idx]:.4f}, Raw: {max_scores[idx]:.4f}") 