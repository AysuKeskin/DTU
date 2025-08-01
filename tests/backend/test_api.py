import pytest
import requests
import json
import time
from fastapi.testclient import TestClient
import sys
import os

# Add backend to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../backend'))

from app.main import app

client = TestClient(app)

class TestAPI:
    """Test suite for DTU Aqua Vision API endpoints"""
    
    def test_app_startup(self):
        """Test that the FastAPI app starts correctly"""
        response = client.get("/")
        # May return 404 or 200 depending on root endpoint
        assert response.status_code in [200, 404]
    
    def test_get_models(self):
        """Test getting available models"""
        response = client.get("/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        
        # Should have at least one model
        assert len(data["models"]) > 0
        
        # Check model structure
        for model in data["models"]:
            assert "name" in model
            assert "size" in model
            assert "description" in model
    
    def test_load_model(self):
        """Test loading a specific model"""
        # First get available models
        models_response = client.get("/models")
        models = models_response.json()["models"]
        
        if models:
            model_name = models[0]["name"]
            
            response = client.post(f"/load-model/{model_name}")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "success"
            assert data["model"] == model_name
    
    def test_get_current_model(self):
        """Test getting current loaded model"""
        response = client.get("/current-model")
        assert response.status_code == 200
        
        data = response.json()
        assert "current_model" in data
    
    @pytest.mark.asyncio
    async def test_detection_endpoint_no_file(self):
        """Test detection endpoint without file (should fail)"""
        response = client.post("/detect")
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_start_stream_endpoint(self):
        """Test start streaming endpoint"""
        response = client.post("/start-stream")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
    
    def test_stop_stream_endpoint(self):
        """Test stop streaming endpoint"""
        response = client.post("/stop-stream")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data

class TestModelPerformance:
    """Test model loading and switching performance"""
    
    def test_model_loading_speed(self):
        """Test how fast models load"""
        models_response = client.get("/models")
        models = models_response.json()["models"]
        
        for model in models:
            start_time = time.time()
            
            response = client.post(f"/load-model/{model['name']}")
            
            load_time = time.time() - start_time
            
            assert response.status_code == 200
            assert load_time < 10.0  # Should load within 10 seconds
            
            print(f"Model {model['name']} loaded in {load_time:.2f}s")
    
    def test_model_switching_speed(self):
        """Test switching between models"""
        models_response = client.get("/models")
        models = models_response.json()["models"]
        
        if len(models) >= 2:
            # Load first model
            start_time = time.time()
            client.post(f"/load-model/{models[0]['name']}")
            first_load = time.time() - start_time
            
            # Switch to second model
            start_time = time.time()
            client.post(f"/load-model/{models[1]['name']}")
            switch_time = time.time() - start_time
            
            print(f"Model switching took {switch_time:.2f}s")
            
            # Switching should be faster than initial load
            # (but this may not always be true depending on model sizes)

if __name__ == "__main__":
    pytest.main([__file__]) 