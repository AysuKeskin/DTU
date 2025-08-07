#!/usr/bin/env python3
"""
Test script to verify confidence thresholds are working
"""
import cv2
import numpy as np
from PIL import Image
from app.onnx_model_simple import SimpleONNXModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_confidence_thresholds():
    print("🧪 Testing confidence thresholds...")
    
    # Load model
    model_path = "yolov8n-seg.onnx"  
    model = SimpleONNXModel(model_path)
    
    # Create test image with a person
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    pil_image = Image.fromarray(test_image)
    
    # Test different confidence thresholds
    thresholds = [0.2, 0.4, 0.6, 0.8]
    
    for conf in thresholds:
        print(f"\n🔍 Testing confidence threshold = {conf}")
        detections = model.predict(
            pil_image, 
            conf_threshold=conf,
            iou_threshold=0.5  # Fixed IoU for this test
        )
        
        print(f"Found {len(detections)} detections")
        if detections:
            print("Top detections:")
            sorted_dets = sorted(detections, key=lambda x: x["confidence"], reverse=True)
            for i, det in enumerate(sorted_dets[:5]):
                print(f"  {i+1}. {det['class_name']}: {det['confidence']:.3f}")
    
    print("\n✅ Confidence threshold test complete!")

if __name__ == "__main__":
    test_confidence_thresholds() 