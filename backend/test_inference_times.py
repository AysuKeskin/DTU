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

async def main():
    # Load model
    model = SimpleONNXModel("yolov8n-seg.onnx")
    
    # Create test image
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    test_img[:, :, 0] = 100  # Red channel
    test_img[:, :, 1] = 150  # Green channel
    test_img[:, :, 2] = 200  # Blue channel
    noise = (np.random.rand(640, 640, 3) * 50).astype(np.uint8)
    test_img = np.clip(test_img + noise, 0, 255)
    
    # Convert to PIL Image for predict method
    pil_img = Image.fromarray(test_img)
    
    logger.info("\n=== Starting Inference Time Comparison ===\n")
    
    # Test parameters
    num_runs = 10
    conf_threshold = 0.25
    iou_threshold = 0.45
    
    # Test predict method
    predict_times = []
    predict_detections = []
    logger.info("Testing predict method...")
    for i in range(num_runs):
        start = time.time()
        detections = model.predict(pil_img, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        elapsed = time.time() - start
        predict_times.append(elapsed)
        predict_detections.append(len(detections))
        logger.info(f"Run {i+1}: {elapsed:.3f}s - {len(detections)} detections")
    
    # Test fast_live_detect_async method
    async_times = []
    async_detections = []
    logger.info("\nTesting fast_live_detect_async method...")
    for i in range(num_runs):
        start = time.time()
        detections = await model.fast_live_detect_async(
            test_img,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=50
        )
        elapsed = time.time() - start
        async_times.append(elapsed)
        async_detections.append(len(detections))
        logger.info(f"Run {i+1}: {elapsed:.3f}s - {len(detections)} detections")
    
    # Calculate statistics
    avg_predict = sum(predict_times) / len(predict_times)
    avg_async = sum(async_times) / len(async_times)
    avg_predict_dets = sum(predict_detections) / len(predict_detections)
    avg_async_dets = sum(async_detections) / len(async_detections)
    
    logger.info("\n=== Results ===")
    logger.info(f"predict method:")
    logger.info(f"  Average time: {avg_predict:.3f}s")
    logger.info(f"  Average detections: {avg_predict_dets:.1f}")
    logger.info(f"  Min time: {min(predict_times):.3f}s")
    logger.info(f"  Max time: {max(predict_times):.3f}s")
    
    logger.info(f"\nfast_live_detect_async method:")
    logger.info(f"  Average time: {avg_async:.3f}s")
    logger.info(f"  Average detections: {avg_async_dets:.1f}")
    logger.info(f"  Min time: {min(async_times):.3f}s")
    logger.info(f"  Max time: {max(async_times):.3f}s")
    
    speedup = avg_predict / avg_async
    logger.info(f"\nSpeedup: {speedup:.2f}x")

if __name__ == "__main__":
    asyncio.run(main()) 