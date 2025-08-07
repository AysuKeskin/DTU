#!/usr/bin/env python3
"""
Test script to verify image detection respects confidence and IoU parameters
"""
import requests
import numpy as np
from PIL import Image
import io

def test_image_detection_params():
    print("🧪 Testing image detection parameter handling...")
    
    # Create a test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    pil_image = Image.fromarray(test_image)
    
    # Save to bytes
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # Test different parameter combinations
    test_cases = [
        {"confidence": 0.3, "iou_threshold": 0.3, "desc": "Low conf, Low IoU"},
        {"confidence": 0.6, "iou_threshold": 0.7, "desc": "High conf, High IoU"}, 
        {"confidence": 0.8, "iou_threshold": 0.2, "desc": "Very high conf, Low IoU"},
    ]
    
    for case in test_cases:
        print(f"\n🔍 Testing: {case['desc']}")
        print(f"   Confidence: {case['confidence']}, IoU: {case['iou_threshold']}")
        
        try:
            # Reset file pointer
            img_bytes.seek(0)
            
            # Prepare form data
            files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
            data = {
                'confidence': str(case['confidence']),
                'iou_threshold': str(case['iou_threshold'])
            }
            
            # Make request
            response = requests.post(
                'http://localhost:8000/detect',
                files=files,
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success: {len(result['detections'])} detections")
                print(f"   Processing time: {result['processing_time']:.3f}s")
                print(f"   Model: {result['model_version']}")
                
                # Show top detections
                if result['detections']:
                    print("   Top detections:")
                    for i, det in enumerate(result['detections'][:3]):
                        print(f"     {i+1}. {det['class_name']}: {det['confidence']:.3f}")
            else:
                print(f"   ❌ Failed: HTTP {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    print("\n✅ Image detection parameter test complete!")

if __name__ == "__main__":
    test_image_detection_params() 