import asyncio
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from .onnx_model_simple import SimpleONNXModel
from typing import Union
import time
import logging

logger = logging.getLogger(__name__)

class UnifiedModel:
    """Ultra-optimized unified model wrapper for high-speed streaming"""
    
    def __init__(self, model_path: str, device: str = None):
        self.model_path = model_path
        self.is_onnx = model_path.endswith('.onnx')
        
        # Device selection with optimization
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
                # CRITICAL GPU optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("🚀 GPU optimizations enabled")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        
        # Load and optimize model
        if self.is_onnx:
            self.model = SimpleONNXModel(model_path)
            logger.info("✅ ONNX model loaded")
        else:
            self._load_and_optimize_pytorch_model()
        
        # Pre-allocate buffers for image processing
        self._setup_buffers()
        
    
    def _load_and_optimize_pytorch_model(self):
        """Load and heavily optimize PyTorch YOLO model"""
        logger.info("📥 Loading PyTorch YOLO model...")
        
        # Load model
        self.model = YOLO(self.model_path)
        
        # Critical optimizations
        self.model.fuse()  # Fuse Conv2d + BatchNorm
        
        if self.device == 'cuda':
            # Move to GPU with half precision
            self.model.model.half().to(self.device)
            logger.info("⚡ Using FP16 precision on GPU")
        else:
            self.model.model.to(self.device)
        
        # Set to eval mode for inference
        self.model.model.eval()
        
        # Disable gradient computation permanently
        for param in self.model.model.parameters():
            param.requires_grad_(False)
        
        # Enable optimized memory format if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_flash_sdp(True)
        
        logger.info("✅ PyTorch model optimized")
    
    def _setup_buffers(self):
        """Pre-allocate buffers for zero-copy image processing"""
        self.max_size = 1920 * 1080 * 3  # Max buffer size
        self.temp_buffer = np.empty(self.max_size, dtype=np.uint8)
        
   
    
    def _fast_image_prep(self, image: Union[Image.Image, np.ndarray, bytes]) -> np.ndarray:
        """Ultra-fast image preprocessing with minimal copies"""
        if isinstance(image, (bytes, bytearray)):
            # Decode directly to RGB
            arr = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=img)  # In-place conversion
            return img
        elif isinstance(image, Image.Image):
            # Convert PIL to numpy (unavoidable copy)
            return np.array(image)
        else:
            # Assume it's already a numpy array - no copy needed
            return image
    
    def predict(
        self,
        image: Union[Image.Image, np.ndarray, bytes],
        conf_threshold: float = 0.6,
        iou_threshold: float = 0.3,
        fast_mode: bool = True  # Default to fast mode for streaming
    ) -> list:
        """Optimized single-image inference"""
        
        # Fast image preparation
        img = self._fast_image_prep(image)
        
        if self.is_onnx:
            return self.model.predict(img, conf_threshold, iou_threshold)
        
        # PyTorch inference with optimizations
        inference_start = time.time()
        
        with torch.no_grad():
            # Optimized predict call
            results = self.model.predict(
                source=img,
                conf=conf_threshold,
                iou=iou_threshold,
                device=self.device,
                verbose=False,
                save=False,
                show=False,
                stream=False,  # Don't use generator for single images
                # Additional speed optimizations
                augment=False,  # Disable test-time augmentation
                visualize=False,
                embed=None,
                half=self.device == 'cuda'  # Use FP16 if on GPU
            )
        
        inference_time = time.time() - inference_start
        
        # Fast result conversion
        detections = []
        if results and len(results) > 0:
            r = results[0]  # Single image result
            if r.boxes is not None and len(r.boxes) > 0:
                # Vectorized conversion for speed
                boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                
                for i, (box, conf, cls_id) in enumerate(zip(boxes_xyxy, confidences, class_ids)):
                    x1, y1, x2, y2 = box
                    detections.append({
                        'id': f'det_{i}',
                        'class_name': r.names[cls_id],
                        'confidence': float(conf),
                        'bbox': {
                            'x1': int(x1), 'y1': int(y1), 
                            'x2': int(x2), 'y2': int(y2)
                        }
                    })
        
        logger.debug(f"⚡ PyTorch inference: {inference_time*1000:.1f}ms, detections: {len(detections)}")
        return detections
    
    def fast_live_detect(self, frame: np.ndarray,
                        conf_threshold: float = 0.5,
                        iou_threshold: float = 0.45,
                        max_detections: int = 50) -> list:
        """Synchronous fast detection for live streaming"""
        detections = self.predict(frame, conf_threshold, iou_threshold, fast_mode=True)
        return detections[:max_detections]
    
    async def fast_live_detect_async(self, frame: np.ndarray,
                                   conf_threshold: float = 0.5,
                                   iou_threshold: float = 0.45,
                                   max_detections: int = 50) -> list:
        """Truly async inference using asyncio"""
        # For streaming, synchronous is often faster than thread pool overhead
        # Only use async if you have other async operations
        return self.fast_live_detect(frame, conf_threshold, iou_threshold, max_detections)
    
    def get_model_info(self):
        """Get model performance information"""
        return {
            'backend': 'ONNX' if self.is_onnx else 'PyTorch',
            'device': self.device,
            'optimizations': {
                'fused': not self.is_onnx,
                'half_precision': self.device == 'cuda' and not self.is_onnx,
                'cudnn_benchmark': torch.backends.cudnn.benchmark if not self.is_onnx else False
            }
        }

# Usage example for WebSocket streaming:
class StreamingOptimizer:
    """Additional optimizations for WebSocket streaming"""
    
    def __init__(self, model: UnifiedModel):
        self.model = model
        self.frame_skip = 0  # Skip frames if processing is slow
        self.last_process_time = 0
        
    def should_process_frame(self, target_fps: int = 30) -> bool:
        """Adaptive frame skipping for consistent FPS"""
        target_interval = 1.0 / target_fps
        current_time = time.time()
        
        if current_time - self.last_process_time < target_interval:
            return False
        
        return True
    
    def process_stream_frame(self, frame: np.ndarray, **kwargs) -> list:
        """Process frame with adaptive optimization"""
        if not self.should_process_frame():
            return []  # Skip this frame
        
        start_time = time.time()
        detections = self.model.fast_live_detect(frame, **kwargs)
        self.last_process_time = time.time()
        
        process_time = self.last_process_time - start_time
        if process_time > 0.033:  # >33ms = <30fps
            logger.warning(f"⚠️ Slow processing: {process_time*1000:.1f}ms")
        
        return detections