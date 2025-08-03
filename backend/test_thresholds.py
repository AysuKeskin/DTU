import asyncio
import time
import cv2
import numpy as np
from PIL import Image
from app.onnx_model_simple import SimpleONNXModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_thresholds():
    # Load model
    model = SimpleONNXModel("yolov8n-seg.onnx")
    
    # Create test image with some objects
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    test_img[:, :, 0] = 100  # Red channel
    test_img[:, :, 1] = 150  # Green channel
    test_img[:, :, 2] = 200  # Blue channel
    noise = (np.random.rand(640, 640, 3) * 50).astype(np.uint8)
    test_img = np.clip(test_img + noise, 0, 255)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(test_img)
    
    logger.info("\n=== Testing Threshold Adjustments ===\n")
    
    # Test different confidence thresholds
    confidence_tests = [0.1, 0.3, 0.5, 0.7, 0.9]
    iou_threshold = 0.45
    
    logger.info("Testing confidence threshold adjustments:")
    for conf in confidence_tests:
        start = time.time()
        detections = model.predict(pil_img, conf_threshold=conf, iou_threshold=iou_threshold)
        elapsed = time.time() - start
        
        logger.info(f"Confidence {conf}: {len(detections)} detections in {elapsed:.3f}s")
        if detections:
            confidences = [d["confidence"] for d in detections]
            logger.info(f"  Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
    
    # Test different IoU thresholds
    logger.info("\nTesting IoU threshold adjustments:")
    confidence = 0.25
    iou_tests = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for iou in iou_tests:
        start = time.time()
        detections = model.predict(pil_img, conf_threshold=confidence, iou_threshold=iou)
        elapsed = time.time() - start
        
        logger.info(f"IoU {iou}: {len(detections)} detections in {elapsed:.3f}s")
    
    # Test async method thresholds
    logger.info("\nTesting async method thresholds:")
    for conf in [0.1, 0.3, 0.5]:
        start = time.time()
        detections = await model.fast_live_detect_async(
            test_img,
            conf_threshold=conf,
            iou_threshold=iou_threshold,
            max_detections=50
        )
        elapsed = time.time() - start
        
        logger.info(f"Async Confidence {conf}: {len(detections)} detections in {elapsed:.3f}s")
        if detections:
            confidences = [d["confidence"] for d in detections]
            logger.info(f"  Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")

if __name__ == "__main__":
    asyncio.run(test_thresholds()) 