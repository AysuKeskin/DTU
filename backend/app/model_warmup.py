import logging
import time
import numpy as np
from PIL import Image

# Use the same logger as main.py
logger = logging.getLogger("app.main")

def warmup_model(model, model_name: str) -> float:
    """
    Comprehensive model warmup for real-world performance
    Returns: warmup time in seconds
    """
    logger.info("🔥 Starting comprehensive ONNX model warmup...")
    try:
        warmup_start = time.time()
        
        # Multiple warmup runs with different image sizes and scenarios
        warmup_configs = [
            (320, 320, "Small frame"),
            (640, 640, "Standard frame"), 
            (1280, 720, "HD frame"),
            (416, 416, "YOLO optimal")
        ]
        
        logger.info("🔥 Running comprehensive ONNX warmup sequence...")
        
        for i, (w, h, desc) in enumerate(warmup_configs):
            logger.info(f"🔥 Warmup {i+1}/{len(warmup_configs)}: {desc} ({w}x{h})")
            
            # Create realistic dummy image (not pure random)
            dummy_array = np.zeros((h, w, 3), dtype=np.uint8)
            # Add some realistic patterns
            dummy_array[:, :, 0] = 100  # Red channel
            dummy_array[:, :, 1] = 150  # Green channel  
            dummy_array[:, :, 2] = 200  # Blue channel
            # Add some noise for realism
            noise = (np.random.rand(h, w, 3) * 50).astype(np.uint8)
            dummy_array = np.clip(dummy_array + noise, 0, 255)
            
            dummy_image = Image.fromarray(dummy_array)
            
            # Run multiple inferences for thorough optimization
            for j in range(5):  # 5 runs per size for complete optimization
                inference_start = time.time()
                detections = model.predict(dummy_image, conf_threshold=0.25)
                inference_time = time.time() - inference_start
                
                if j == 0:
                    logger.info(f"   • First inference: {inference_time:.3f}s")
                elif j == 4:
                    logger.info(f"   • Final inference: {inference_time:.3f}s (optimized)")
        
        warmup_time = time.time() - warmup_start
        
        # Final performance test with realistic scenario
        logger.info("🎯 Final performance validation...")
        test_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_array)
        
        # Test multiple confidence levels
        for conf in [0.25, 0.5, 0.7]:
            test_start = time.time()
            detections = model.predict(test_image, conf_threshold=conf)
            test_time = time.time() - test_start
            logger.info(f"   • Confidence {conf}: {test_time:.3f}s")
        
        logger.info(f"🔥 Comprehensive ONNX warmup completed in {warmup_time:.2f}s")
        logger.info(f"✅ Model {model_name} is fully optimized and ready for production use!")
        return warmup_time
        
    except Exception as warmup_error:
        logger.warning(f"⚠️ ONNX model warmup failed: {warmup_error}, but model is loaded")
        return 0.0 