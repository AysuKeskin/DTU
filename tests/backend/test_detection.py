import pytest
import numpy as np
import cv2
import time
import io
import base64
from PIL import Image
import sys
import os

# Add backend to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../backend'))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestDetectionAccuracy:
    """Test detection accuracy and consistency"""
    
    def create_test_image(self, width=320, height=240, color=(255, 0, 0)):
        """Create a simple test image"""
        # Create a red rectangle on white background
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (50, 50), (width-50, height-50), color, -1)
        
        # Convert to PIL Image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        pil_img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        return img_bytes.getvalue()
    
    def test_detection_with_simple_image(self):
        """Test detection with a simple generated image"""
        # Load a model first
        models_response = client.get("/models")
        if models_response.status_code == 200:
            models = models_response.json()["models"]
            if models:
                client.post(f"/load-model/{models[0]['name']}")
        
        # Create test image
        test_image = self.create_test_image()
        
        # Send for detection
        files = {"file": ("test.jpg", test_image, "image/jpeg")}
        response = client.post("/detect", files=files)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "detections" in data
        assert "processing_time" in data
        assert "model_version" in data
        
        # Check detection structure
        for detection in data["detections"]:
            assert "id" in detection
            assert "class_name" in detection
            assert "confidence" in detection
            assert "bbox" in detection
            
            # Confidence should be between 0 and 1
            assert 0.0 <= detection["confidence"] <= 1.0
            
            # Bbox should have valid coordinates
            bbox = detection["bbox"]
            assert "x1" in bbox and "y1" in bbox
            assert "x2" in bbox and "y2" in bbox
            assert bbox["x1"] < bbox["x2"]
            assert bbox["y1"] < bbox["y2"]
    
    def test_detection_performance_different_sizes(self):
        """Test detection performance with different image sizes"""
        sizes = [
            (160, 120),   # Very small
            (320, 240),   # Current size
            (640, 480),   # Larger
        ]
        
        # Load a model first
        models_response = client.get("/models")
        if models_response.status_code == 200:
            models = models_response.json()["models"]
            if models:
                client.post(f"/load-model/{models[0]['name']}")
        
        performance_results = {}
        
        for width, height in sizes:
            test_image = self.create_test_image(width, height)
            
            # Time the detection
            start_time = time.time()
            
            files = {"file": ("test.jpg", test_image, "image/jpeg")}
            response = client.post("/detect", files=files)
            
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                processing_time = data.get("processing_time", end_time - start_time)
                performance_results[f"{width}x{height}"] = processing_time
                
                print(f"Detection at {width}x{height}: {processing_time:.3f}s")
                
                # Smaller images should generally be faster
                # (though this isn't guaranteed due to model behavior)
        
        return performance_results
    
    def test_detection_consistency(self):
        """Test that detection gives consistent results for the same image"""
        # Load a model first
        models_response = client.get("/models")
        if models_response.status_code == 200:
            models = models_response.json()["models"]
            if models:
                client.post(f"/load-model/{models[0]['name']}")
        
        test_image = self.create_test_image()
        files = {"file": ("test.jpg", test_image, "image/jpeg")}
        
        # Run detection multiple times
        results = []
        for i in range(3):
            response = client.post("/detect", files=files)
            if response.status_code == 200:
                data = response.json()
                results.append(data)
        
        # Check that results are reasonably consistent
        if len(results) >= 2:
            # Compare number of detections
            detection_counts = [len(r["detections"]) for r in results]
            
            # Should be consistent (allowing for some variation)
            assert max(detection_counts) - min(detection_counts) <= 1
            
            print(f"Detection consistency: {detection_counts}")

class TestDetectionPerformance:
    """Test detection speed and efficiency"""
    
    def test_detection_speed_benchmark(self):
        """Benchmark detection speed"""
        # Load a model first
        models_response = client.get("/models")
        if models_response.status_code == 200:
            models = models_response.json()["models"]
            if models:
                client.post(f"/load-model/{models[0]['name']}")
        
        # Create test image
        test_image_data = TestDetectionAccuracy().create_test_image()
        files = {"file": ("test.jpg", test_image_data, "image/jpeg")}
        
        # Run multiple detections and measure time
        times = []
        num_runs = 5
        
        for i in range(num_runs):
            start_time = time.time()
            
            response = client.post("/detect", files=files)
            
            end_time = time.time()
            
            if response.status_code == 200:
                times.append(end_time - start_time)
        
        if times:
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            print(f"Detection benchmark ({num_runs} runs):")
            print(f"  Average: {avg_time:.3f}s")
            print(f"  Min: {min_time:.3f}s")
            print(f"  Max: {max_time:.3f}s")
            
            # Assert reasonable performance (adjust threshold as needed)
            assert avg_time < 2.0  # Should be under 2 seconds on average
            
            return {
                "average": avg_time,
                "min": min_time,
                "max": max_time,
                "runs": num_runs
            }
    
    def test_model_comparison(self):
        """Compare performance between different models"""
        models_response = client.get("/models")
        if models_response.status_code != 200:
            pytest.skip("Cannot get models list")
        
        models = models_response.json()["models"]
        if len(models) < 2:
            pytest.skip("Need at least 2 models for comparison")
        
        test_image_data = TestDetectionAccuracy().create_test_image()
        files = {"file": ("test.jpg", test_image_data, "image/jpeg")}
        
        model_performance = {}
        
        for model in models:
            # Load model
            load_response = client.post(f"/load-model/{model['name']}")
            if load_response.status_code != 200:
                continue
            
            # Test detection speed
            start_time = time.time()
            response = client.post("/detect", files=files)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                model_performance[model['name']] = {
                    "time": end_time - start_time,
                    "detections": len(data["detections"]),
                    "processing_time": data.get("processing_time", 0)
                }
        
        # Print comparison
        print("\nModel Performance Comparison:")
        for model_name, perf in model_performance.items():
            print(f"  {model_name}: {perf['time']:.3f}s, {perf['detections']} detections")
        
        return model_performance

if __name__ == "__main__":
    pytest.main([__file__]) 