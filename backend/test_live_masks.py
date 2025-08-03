import asyncio
import cv2
import numpy as np
from PIL import Image
from app.onnx_model_simple import SimpleONNXModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_live_mask_generation():
    """Test if masks are generated in live streaming mode"""
    
    model = SimpleONNXModel("yolov8n-seg.onnx")
    
    # Create a test image with some realistic content
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Create a simple scene - add some rectangles to simulate objects
    test_img[100:300, 100:300] = [100, 150, 200]  # Blue-ish rectangle
    test_img[350:500, 350:500] = [200, 100, 150]  # Purple-ish rectangle
    test_img[200:400, 450:600] = [150, 200, 100]  # Green-ish rectangle
    
    # Add some noise for realism
    noise = (np.random.rand(640, 640, 3) * 50).astype(np.uint8)
    test_img = np.clip(test_img + noise, 0, 255)
    
    logger.info("🧪 Testing live mask generation...")
    
    # Test with different confidence thresholds
    for conf in [0.1, 0.3, 0.5]:
        logger.info(f"\n🎯 Testing with confidence threshold: {conf}")
        
        detections = await model.fast_live_detect_async(
            test_img,
            conf_threshold=conf,
            iou_threshold=0.45,
            max_detections=50
        )
        
        logger.info(f"Found {len(detections)} detections")
        
        for i, detection in enumerate(detections[:5]):  # Show first 5
            mask_info = f"mask: {len(detection['mask'])} points" if detection.get('mask') else "no mask"
            logger.info(f"  Detection {i}: {detection['class_name']} ({detection['confidence']:.3f}) - {mask_info}")
            
            if detection.get('mask'):
                logger.info(f"    First 3 mask points: {detection['mask'][:3]}")
    
    # Test the synchronous version for comparison
    logger.info(f"\n🔄 Testing synchronous predict method for comparison...")
    pil_img = Image.fromarray(test_img)
    sync_detections = model.predict(pil_img, conf_threshold=0.3, iou_threshold=0.45)
    
    logger.info(f"Sync method found {len(sync_detections)} detections")
    sync_mask_count = sum(1 for det in sync_detections if det.get('mask'))
    logger.info(f"Sync method generated {sync_mask_count} masks")

if __name__ == "__main__":
    asyncio.run(test_live_mask_generation()) 