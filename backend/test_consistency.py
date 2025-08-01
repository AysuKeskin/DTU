#!/usr/bin/env python3
"""
Test script to check ONNX model consistency
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from onnx_model_simple import SimpleONNXModel
from PIL import Image
import numpy as np

def test_consistency():
    """Test if the ONNX model gives consistent results"""
    
    # Use a real test image
    image_path = "person_test.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Test image {image_path} not found!")
        return False
    
    pil_image = Image.open(image_path).convert("RGB")
    print(f"✅ Loaded test image: {image_path} ({pil_image.size})")
    
    # Load the model
    model = SimpleONNXModel("yolov8n-seg.onnx")
    
    print("Testing consistency with real person image...")
    
    # Run multiple predictions on the same image
    results = []
    for i in range(5):
        detections = model.predict(pil_image, conf_threshold=0.01)  # Very low threshold to see any detections
        results.append(detections)
        print(f"Run {i+1}: {len(detections)} detections")
        
        # Print confidence scores for first detection if any
        if detections:
            print(f"  First detection confidence: {detections[0]['confidence']:.4f}")
            print(f"  First detection class: {detections[0]['class_name']}")
        else:
            print("  No detections found")
    
    # Check if results are consistent
    first_result = results[0]
    consistent = True
    
    for i, result in enumerate(results[1:], 1):
        if len(result) != len(first_result):
            print(f"❌ Inconsistent number of detections: {len(first_result)} vs {len(result)}")
            consistent = False
            break
            
        for j, (first_det, det) in enumerate(zip(first_result, result)):
            if abs(first_det['confidence'] - det['confidence']) > 0.001:
                print(f"❌ Inconsistent confidence at detection {j}: {first_det['confidence']:.4f} vs {det['confidence']:.4f}")
                consistent = False
                break
    
    if consistent:
        print("✅ Results are consistent!")
    else:
        print("❌ Results are inconsistent!")
    
    return consistent

if __name__ == "__main__":
    test_consistency() 