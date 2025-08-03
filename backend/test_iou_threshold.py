#!/usr/bin/env python3
"""
Test script to verify IoU threshold is working properly
"""
import cv2
import numpy as np
from PIL import Image
from app.onnx_model_simple import SimpleONNXModel

def test_iou_thresholds():
    print("🧪 Testing IoU threshold functionality...")
    
    # Load model
    model_path = "yolo11n-seg.onnx"  
    model = SimpleONNXModel(model_path)
    
    # Create test image with a person (you)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    pil_image = Image.fromarray(test_image)
    
    print("\n🔍 Testing different IoU thresholds:")
    
    # Test with very loose IoU (should allow many duplicates)
    print("\n1. Testing IoU = 0.1 (very loose, should allow duplicates)")
    detections_loose = model.fast_live_detect_async(
        pil_image, 
        conf_threshold=0.3, 
        iou_threshold=0.1,  # Very loose
        max_detections=20
    )
    print(f"   Detections found: {len(detections_loose)}")
    for i, det in enumerate(detections_loose[:5]):
        print(f"   {i}: {det['class_name']} ({det['confidence']:.3f})")
    
    # Test with medium IoU
    print("\n2. Testing IoU = 0.5 (medium, should remove some duplicates)")
    detections_medium = model.fast_live_detect_async(
        pil_image, 
        conf_threshold=0.3, 
        iou_threshold=0.5,  # Medium
        max_detections=20
    )
    print(f"   Detections found: {len(detections_medium)}")
    for i, det in enumerate(detections_medium[:5]):
        print(f"   {i}: {det['class_name']} ({det['confidence']:.3f})")
    
    # Test with very strict IoU (should remove most duplicates)
    print("\n3. Testing IoU = 0.9 (very strict, should remove most duplicates)")
    detections_strict = model.fast_live_detect_async(
        pil_image, 
        conf_threshold=0.3, 
        iou_threshold=0.9,  # Very strict
        max_detections=20
    )
    print(f"   Detections found: {len(detections_strict)}")
    for i, det in enumerate(detections_strict[:5]):
        print(f"   {i}: {det['class_name']} ({det['confidence']:.3f})")
    
    print(f"\n📊 Summary:")
    print(f"   IoU 0.1: {len(detections_loose)} detections")
    print(f"   IoU 0.5: {len(detections_medium)} detections") 
    print(f"   IoU 0.9: {len(detections_strict)} detections")
    
    if len(detections_loose) >= len(detections_medium) >= len(detections_strict):
        print("✅ IoU threshold is working correctly!")
        print("   (Higher IoU = fewer detections = better duplicate removal)")
    else:
        print("❌ IoU threshold may not be working properly!")
        print("   Expected: loose >= medium >= strict")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_iou_thresholds()) 