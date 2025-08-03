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

async def test_threshold_connection():
    """Test if thresholds are properly connected and adjustable"""
    
    # Load model
    model = SimpleONNXModel("yolov8n-seg.onnx")
    
    # Create test image
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    test_img[:, :, 0] = 100  # Red channel
    test_img[:, :, 1] = 150  # Green channel
    test_img[:, :, 2] = 200  # Blue channel
    noise = (np.random.rand(640, 640, 3) * 50).astype(np.uint8)
    test_img = np.clip(test_img + noise, 0, 255)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(test_img)
    
    logger.info("\n=== Testing Threshold Connection ===\n")
    
    # Test 1: Verify that different confidence thresholds produce different results
    logger.info("Test 1: Confidence threshold adjustment")
    conf_tests = [0.1, 0.3, 0.5, 0.7, 0.9]
    iou_fixed = 0.45
    
    for conf in conf_tests:
        detections = model.predict(pil_img, conf_threshold=conf, iou_threshold=iou_fixed)
        logger.info(f"Confidence {conf}: {len(detections)} detections")
        if detections:
            confidences = [d["confidence"] for d in detections]
            logger.info(f"  Detection confidences: {min(confidences):.3f} - {max(confidences):.3f}")
    
    # Test 2: Verify that different IoU thresholds produce different results
    logger.info("\nTest 2: IoU threshold adjustment")
    conf_fixed = 0.25
    iou_tests = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for iou in iou_tests:
        detections = model.predict(pil_img, conf_threshold=conf_fixed, iou_threshold=iou)
        logger.info(f"IoU {iou}: {len(detections)} detections")
    
    # Test 3: Verify async method threshold adjustment
    logger.info("\nTest 3: Async method threshold adjustment")
    for conf in [0.1, 0.3, 0.5]:
        detections = await model.fast_live_detect_async(
            test_img,
            conf_threshold=conf,
            iou_threshold=iou_fixed,
            max_detections=50
        )
        logger.info(f"Async Confidence {conf}: {len(detections)} detections")
    
    # Test 4: Check if thresholds are being applied correctly
    logger.info("\nTest 4: Threshold application verification")
    
    # Test with very high confidence - should get few or no detections
    high_conf_detections = model.predict(pil_img, conf_threshold=0.9, iou_threshold=0.45)
    logger.info(f"High confidence (0.9): {len(high_conf_detections)} detections")
    
    # Test with very low confidence - should get more detections
    low_conf_detections = model.predict(pil_img, conf_threshold=0.1, iou_threshold=0.45)
    logger.info(f"Low confidence (0.1): {len(low_conf_detections)} detections")
    
    # Verify that low confidence gets more detections than high confidence
    if len(low_conf_detections) >= len(high_conf_detections):
        logger.info("✅ Threshold adjustment working correctly")
    else:
        logger.warning("⚠️ Threshold adjustment may not be working correctly")
    
    # Test 5: Check default values
    logger.info("\nTest 5: Default values check")
    logger.info("Frontend defaults:")
    logger.info("  Confidence: 0.3 (30%)")
    logger.info("  IoU: 0.2 (20%)")
    
    logger.info("Backend defaults:")
    logger.info("  /detect endpoint: confidence=0.6, iou_threshold=0.7")
    logger.info("  /ws endpoint: confidence=0.6, iou_threshold=0.6")
    logger.info("  /ws-video endpoint: confidence=0.3, iou_threshold=0.2")
    logger.info("  predict method: conf_threshold=0.25, iou_threshold=0.3")
    logger.info("  fast_live_detect_async: conf_threshold=0.2, iou_threshold=0.4")

if __name__ == "__main__":
    asyncio.run(test_threshold_connection()) 