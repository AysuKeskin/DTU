import logging
import time
from typing import List, Dict, Union, Optional
import numpy as np
import cv2
from PIL import Image
import torch
from ultralytics import YOLO
import onnxruntime as ort

logger = logging.getLogger("app.main")

class UnifiedModel:
    """Unified wrapper for both ONNX and PyTorch YOLO models"""
    
    def __init__(self, model_path: str):
        """Initialize model wrapper
        
        Args:
            model_path: Path to either .pt or .onnx model file
        """
        self.model_path = model_path
        self.model_type = 'pt' if model_path.endswith('.pt') else 'onnx'
        logger.info(f"Loading {self.model_type.upper()} model from {model_path}")
        
        # COCO class names (hardcoded for performance)
        self.class_names = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
            6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
            11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
            16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant",
            21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella",
            26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis",
            31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
            36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
            41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
            46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
            51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
            56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
            61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote",
            66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster",
            71: "sink", 72: "refrigerator", 73: "book", 74: "clock", 75: "vase",
            76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
        }
        
        if self.model_type == 'pt':
            # Load PyTorch model
            self.model = YOLO(model_path)
            logger.info("✅ PyTorch model loaded successfully")
            
        else:
            # Load ONNX model
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Check available providers
            available_providers = ort.get_available_providers()
            logger.info(f"Available ONNX providers: {available_providers}")
            
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                logger.info("🚀 Using CUDA GPU acceleration for ONNX inference")
            elif 'CoreMLExecutionProvider' in available_providers:
                providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
                logger.info("🚀 Using Apple Silicon GPU acceleration for ONNX inference")
            else:
                providers = ['CPUExecutionProvider']
                logger.info("⚠️ Using CPU only - no GPU acceleration available")
            
            try:
                self.model = ort.InferenceSession(model_path, session_options, providers=providers)
                
                # Get input details
                inp = self.model.get_inputs()[0]
                self.input_name = inp.name
                
                # Handle dynamic dimensions
                def get_dim(d):
                    if isinstance(d, (int, float)):
                        return int(d)
                    if isinstance(d, str):
                        if d.lower() in ['batch', 'n']:
                            return 1
                        if d.isdigit():
                            return int(d)
                        return 640  # Default size for dynamic H,W dimensions
                    return 640  # Default fallback
                
                # Convert shape dimensions safely
                raw_shape = inp.shape
                self.input_shape = tuple(get_dim(d) for d in raw_shape)
                logger.info(f"Model input shape: {raw_shape} -> {self.input_shape}")
                
                # Pre-allocate buffers
                self.input_buffer = np.empty(self.input_shape, dtype=np.float32)
                self.max_detections = 50
                self.coord_buffer = np.empty((self.max_detections, 4), dtype=np.float32)
                
                # Get mask dimensions if available
                outputs = self.model.get_outputs()
                if len(outputs) > 1:
                    nm = get_dim(outputs[1].shape[1])
                    self.mask_coeff_buffer = np.empty((self.max_detections, nm), dtype=np.float32)
                    logger.info(f"Mask prototypes: {nm}")
                else:
                    self.mask_coeff_buffer = None
                
                # Set up IO binding
                self.io_binding = self.model.io_binding()
                logger.info("✅ ONNX model loaded successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to load ONNX model: {str(e)}")
                raise
    
    def predict(self, image: Union[np.ndarray, Image.Image], conf_threshold: float = 0.6, 
               iou_threshold: float = 0.3, max_detections: int = 50) -> List[Dict]:
        """Run inference on an image
        
        Args:
            image: Input image (numpy array or PIL Image)
            conf_threshold: Confidence threshold (0.0-1.0)
            iou_threshold: IoU threshold for NMS (0.0-1.0)
            max_detections: Maximum number of detections to return
            
        Returns:
            List of detection dictionaries
        """
        try:
            # Convert image if needed
            if isinstance(image, Image.Image):
                img = np.array(image)
                logger.info(f"🖼️ PIL Image input: {img.shape}")
            elif isinstance(image, np.ndarray):
                img = image
                logger.info(f"📊 NumPy array input: {img.shape}")
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            original_h, original_w = img.shape[:2]
            
            # Run inference based on model type
            if self.model_type == 'pt':
                # PyTorch inference
                results = self.model(img, conf=conf_threshold, iou=iou_threshold, max_det=max_detections)
                result = results[0]  # Get first image result
                
                # Convert to common format
                detections = []
                for i, (box, mask) in enumerate(zip(result.boxes, result.masks if hasattr(result, 'masks') else [])):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    detection = {
                        "id": f"det_{i}",
                        "class_name": self.class_names.get(cls_id, f"class_{cls_id}"),
                        "confidence": conf,
                        "bbox": {
                            "x1": int(x1), "y1": int(y1),
                            "x2": int(x2), "y2": int(y2)
                        },
                        "mask": None,
                        "image_size": {"width": original_w, "height": original_h}
                    }
                    
                    # Add mask if available
                    if hasattr(result, 'masks') and mask is not None:
                        try:
                            mask_data = mask.data[0].cpu().numpy()
                            mask_binary = (mask_data > 0.5).astype(np.uint8) * 255
                            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                largest = max(contours, key=cv2.contourArea)
                                detection["mask"] = largest.reshape(-1, 2).tolist()
                        except Exception as e:
                            logger.warning(f"Mask generation failed: {e}")
                    
                    detections.append(detection)
                
                return detections
                
            else:
                # ONNX inference (use existing ONNX code)
                img_resized = cv2.resize(img, (640, 640))
                img_norm = img_resized.astype(np.float32) / 255.0
                img_batch = np.transpose(img_norm, (2, 0, 1))[None, ...]
                
                # Run inference
                inference_start = time.time()
                outputs = self.model.run(None, {self.input_name: img_batch})
                inference_time = time.time() - inference_start
                logger.info(f"⚡ ONNX inference time: {inference_time:.3f}s")
                
                # Process outputs (use existing ONNX processing code)
                detections = outputs[0][0].T
                boxes = detections[:, :4]
                class_scores = detections[:, 4:84]
                mask_coeffs_raw = detections[:, 84:]
                mask_protos = outputs[1][0] if len(outputs) > 1 else None
                
                # Calculate confidences
                class_scores_sigmoid = 1 / (1 + np.exp(-class_scores))
                confidences = np.max(class_scores_sigmoid, axis=1)
                class_ids = np.argmax(class_scores_sigmoid, axis=1)
                
                # Log confidence distribution
                logger.info("🔍 CONFIDENCE SCORE ANALYSIS:")
                logger.info(f"   Range: {np.min(confidences):.6f} to {np.max(confidences):.6f}")
                logger.info(f"   Mean: {np.mean(confidences):.6f}")
                logger.info(f"   Median: {np.median(confidences):.6f}")
                logger.info(f"   Below 0.5: {np.sum(confidences < 0.5)} ({np.sum(confidences < 0.5)/len(confidences)*100:.2f}%)")
                
                # Filter by confidence
                valid_indices = confidences > conf_threshold
                if not valid_indices.any():
                    return []
                
                valid_boxes = boxes[valid_indices]
                valid_confidences = confidences[valid_indices]
                valid_class_ids = class_ids[valid_indices]
                valid_mask_coeffs = mask_coeffs_raw[valid_indices] if mask_coeffs_raw is not None else None
                
                # Convert coordinates
                x_center, y_center, width, height = valid_boxes.T
                scale_x, scale_y = original_w / 640, original_h / 640
                
                x1 = ((x_center - width / 2) * scale_x).astype(int)
                y1 = ((y_center - height / 2) * scale_y).astype(int)
                x2 = ((x_center + width / 2) * scale_x).astype(int)
                y2 = ((y_center + height / 2) * scale_y).astype(int)
                
                # Apply NMS
                boxes_for_nms = np.column_stack([x1, y1, x2, y2])
                indices = cv2.dnn.NMSBoxes(
                    boxes_for_nms.tolist(),
                    valid_confidences.tolist(),
                    score_threshold=0.0001,
                    nms_threshold=iou_threshold
                )
                
                if len(indices) == 0:
                    return []
                
                indices = indices.flatten()
                sorted_indices = sorted(indices, key=lambda i: valid_confidences[i], reverse=True)[:max_detections]
                
                # Create detections
                detections = []
                for i in sorted_indices:
                    detection = {
                        "id": f"det_{i}",
                        "class_name": self.class_names.get(valid_class_ids[i], f"class_{valid_class_ids[i]}"),
                        "confidence": float(valid_confidences[i]),
                        "bbox": {
                            "x1": int(x1[i]), "y1": int(y1[i]),
                            "x2": int(x2[i]), "y2": int(y2[i])
                        },
                        "mask": None,
                        "image_size": {"width": original_w, "height": original_h}
                    }
                    
                    # Generate mask if available
                    if valid_mask_coeffs is not None and mask_protos is not None:
                        try:
                            mask = np.dot(valid_mask_coeffs[i], mask_protos.reshape(-1, mask_protos.shape[-2] * mask_protos.shape[-1]))
                            mask = mask.reshape(mask_protos.shape[-2:])
                            mask = 1 / (1 + np.exp(-mask))
                            mask = cv2.resize(mask, (original_w, original_h)) > 0.5
                            
                            contours, _ = cv2.findContours(
                                (mask * 255).astype(np.uint8),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE
                            )
                            
                            if contours:
                                largest = max(contours, key=cv2.contourArea)
                                detection["mask"] = largest.reshape(-1, 2).tolist()
                        except Exception as e:
                            logger.warning(f"Mask generation failed: {e}")
                    
                    detections.append(detection)
                
                return detections
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return []
    
    async def predict_stream(self, frame: np.ndarray, conf_threshold: float = 0.6,
                           iou_threshold: float = 0.3, max_detections: int = 50) -> List[Dict]:
        """Async prediction for streaming
        
        Args:
            frame: Input frame as numpy array
            conf_threshold: Confidence threshold (0.0-1.0)
            iou_threshold: IoU threshold for NMS (0.0-1.0)
            max_detections: Maximum number of detections to return
            
        Returns:
            List of detection dictionaries
        """
        # For now, just call predict() - we can optimize this later if needed
        return self.predict(frame, conf_threshold, iou_threshold, max_detections) 