#!/usr/bin/env python3
"""
Compare performance between YOLO11n and YOLO8n ONNX models
"""

import time
from ultralytics import YOLO
import numpy as np
import cv2
import requests
from PIL import Image
import io

def download_test_image():
    """Download a test image"""
    url = "https://ultralytics.com/images/bus.jpg"
    response = requests.get(url)
    image = Image.open(io.BytesIO(response.content))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def test_model(model_name, num_runs=10):
    """Test a model's performance"""
    print(f"\n🔄 Testing {model_name}...")
    
    # Load model
    model = YOLO(model_name)
    
    # Get test image
    image = download_test_image()
    
    # Warmup run
    _ = model(image)
    
    # Time multiple runs
    times = []
    detections = []
    
    for i in range(num_runs):
        start = time.time()
        results = model(image)
        end = time.time()
        
        inference_time = (end - start) * 1000  # Convert to ms
        times.append(inference_time)
        
        # Count detections
        if len(results) > 0:
            num_detections = len(results[0].boxes)
            detections.append(num_detections)
        
        print(f"Run {i+1}: {inference_time:.1f}ms, {num_detections} detections")
    
    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_detections = np.mean(detections)
    
    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'avg_detections': avg_detections,
        'times': times,
        'detections': detections
    }

def compare_models():
    """Compare YOLO11n and YOLO8n"""
    print("🚀 Comparing YOLO11n vs YOLO8n ONNX models...")
    
    # Test both models
    yolo11n_results = test_model("backend/yolo11n-seg.onnx")
    yolo8n_results = test_model("backend/yolov8n-seg.onnx")
    
    # Print comparison
    print("\n📊 Performance Comparison:")
    print("-" * 50)
    print(f"{'Metric':<20} {'YOLO11n':<15} {'YOLO8n':<15}")
    print("-" * 50)
    print(f"{'Average Time (ms)':<20} {yolo11n_results['avg_time']:<15.1f} {yolo8n_results['avg_time']:<15.1f}")
    print(f"{'Std Dev Time (ms)':<20} {yolo11n_results['std_time']:<15.1f} {yolo8n_results['std_time']:<15.1f}")
    print(f"{'Avg Detections':<20} {yolo11n_results['avg_detections']:<15.1f} {yolo8n_results['avg_detections']:<15.1f}")
    
    # Calculate speed difference
    speed_diff = ((yolo8n_results['avg_time'] - yolo11n_results['avg_time']) / yolo8n_results['avg_time']) * 100
    
    print("\n🏃 Speed Comparison:")
    if speed_diff > 0:
        print(f"YOLO11n is {abs(speed_diff):.1f}% faster than YOLO8n")
    else:
        print(f"YOLO8n is {abs(speed_diff):.1f}% faster than YOLO11n")
    
    # Detection comparison
    det_diff = yolo11n_results['avg_detections'] - yolo8n_results['avg_detections']
    print(f"\n🎯 Detection Comparison:")
    if det_diff == 0:
        print("Both models detect the same number of objects")
    else:
        print(f"Difference in detections: {abs(det_diff):.1f} objects")
    
    return yolo11n_results, yolo8n_results

if __name__ == "__main__":
    compare_models() 