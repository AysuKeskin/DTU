#!/usr/bin/env python3
"""
Test script to verify new confidence calculation
"""
import cv2
import numpy as np
from PIL import Image
from app.onnx_model_simple import SimpleONNXModel

def test_confidence_calculation():
    print("🧪 Testing new confidence calculation...")
    
    # Load model
    model_path = "yolo11n-seg.onnx"  
    model = SimpleONNXModel(model_path)
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    pil_image = Image.fromarray(test_image)
    
    print("\n🔍 Testing different confidence thresholds:")
    
    thresholds = [0.3, 0.5, 0.7]
    for conf in thresholds:
        print(f"\nTesting confidence threshold = {conf}")
        detections = model.fast_live_detect_async(
            pil_image, 
            conf_threshold=conf,
            iou_threshold=0.5,  # Fixed IoU for this test
            max_detections=20
        )
        
        print(f"Detections found: {len(detections)}")
        if detections:
            print("Top 5 detections by confidence:")
            sorted_dets = sorted(detections, key=lambda x: x["confidence"], reverse=True)
            for i, det in enumerate(sorted_dets[:5]):
                print(f"  {i+1}. {det['class_name']}: {det['confidence']:.3f}")
        else:
            print("No detections above threshold")
    
    print("\n📊 Testing very low confidence for distribution analysis")
    detections = model.fast_live_detect_async(
        pil_image,
        conf_threshold=0.1,  # Very low to see distribution
        iou_threshold=0.5,
        max_detections=100  # Higher limit to see more
    )
    
    if detections:
        confidences = [d["confidence"] for d in detections]
        print(f"\nConfidence distribution (n={len(confidences)}):")
        print(f"  Min: {min(confidences):.3f}")
        print(f"  Max: {max(confidences):.3f}")
        print(f"  Mean: {np.mean(confidences):.3f}")
        print(f"  Median: {np.median(confidences):.3f}")
        
        # Count by ranges
        ranges = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
        for low, high in ranges:
            count = sum(1 for c in confidences if low <= c < high)
            print(f"  {low:.1f}-{high:.1f}: {count:3d} detections")
    
    print("\n✨ Confidence calculation test complete!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_confidence_calculation()) 