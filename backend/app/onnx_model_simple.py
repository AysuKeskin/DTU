import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple
import logging
import asyncio

logger = logging.getLogger(__name__)

class SimpleONNXModel:
    """High-performance direct ONNX model for fast live detection"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        
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
        
        # GPU OPTIMIZATION: Use CUDA if available, fallback to CPU
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Check available providers and prioritize GPU
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
        
        self.session = ort.InferenceSession(model_path, session_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # PERFORMANCE OPTIMIZATION: Pre-allocate buffers for zero-copy
        self.input_buffer = np.empty((1, 3, 640, 640), dtype=np.float32)
        self.io_binding = self.session.io_binding()
        
        # Pre-allocate coordinate arrays for reuse
        self.coord_buffer = np.empty((50, 4), dtype=np.float32)  # Max 50 detections
        
        logger.info(f"✅ High-performance ONNX model loaded: {model_path}")
        logger.info(f"🚀 Input shape: {self.input_shape}")
        logger.info(f"🎯 Using providers: {self.session.get_providers()}")
        logger.info(f"⚡ Pre-allocated buffers for zero-copy inference")
    
    async def fast_live_detect_async(self, image, conf_threshold: float = 0.2, max_detections: int = 50) -> List[dict]:
        """
        ASYNC ULTRA-FAST live detection optimized for real-time streaming
        
        Optimizations:
        - Async inference (non-blocking WebSocket)
        - GPU acceleration (CUDA/CoreML)
        - IOBinding for zero-copy
        - Pre-allocated buffers
        - Thresholding before sigmoid
        - No mask generation
        - Simplified NMS
        """
        try:
            # Handle input types
            if isinstance(image, Image.Image):
                img = np.array(image)
            elif isinstance(image, np.ndarray):
                img = image
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            original_h, original_w = img.shape[:2]
            
            # FAST PREPROCESSING: Write directly to pre-allocated buffer
            img_resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
            img_norm = img_resized.astype(np.float32) / 255.0
            np.copyto(self.input_buffer[0], np.transpose(img_norm, (2, 0, 1)))
            
            # ASYNC GPU INFERENCE: Non-blocking WebSocket thread
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None, 
                self._run_inference_optimized, 
                self.input_buffer
            )
            
            # FAST POSTPROCESSING: Extract only what we need
            detections = outputs[0][0].T
            boxes = detections[:, :4]
            class_scores = detections[:, 4:84]
            
            # OPTIMIZATION: Threshold before sigmoid (avoid exp on large arrays)
            max_scores = np.max(class_scores, axis=1)
            
            # Convert confidence threshold to logit threshold
            logit_threshold = -np.log(1/conf_threshold - 1) if conf_threshold > 0 and conf_threshold < 1 else 0
            valid_indices = max_scores > logit_threshold
            
            if np.sum(valid_indices) == 0:
                return []
            
            # Apply sigmoid only to valid detections
            valid_scores = max_scores[valid_indices]
            confidences_sigmoid = 1 / (1 + np.exp(-valid_scores))
            
            # LIMIT DETECTIONS: Keep only top N for speed
            if len(confidences_sigmoid) > max_detections:
                top_indices = np.argsort(confidences_sigmoid)[-max_detections:]
                temp_indices = np.where(valid_indices)[0]
                valid_indices = np.zeros_like(valid_indices, dtype=bool)
                valid_indices[temp_indices[top_indices]] = True
                confidences_sigmoid = confidences_sigmoid[top_indices]
            
            valid_boxes = boxes[valid_indices]
            valid_class_scores = class_scores[valid_indices]
            
            # Get class IDs
            class_ids = np.argmax(valid_class_scores, axis=1)
            
            # FAST COORDINATE CONVERSION: Use pre-allocated buffer
            x_center, y_center, width, height = valid_boxes.T
            scale_x, scale_y = original_w / 640, original_h / 640
            
            # Write to pre-allocated coordinate buffer
            self.coord_buffer[:len(valid_boxes), 0] = np.clip(((x_center - width / 2) * scale_x), 0, original_w)
            self.coord_buffer[:len(valid_boxes), 1] = np.clip(((y_center - height / 2) * scale_y), 0, original_h)
            self.coord_buffer[:len(valid_boxes), 2] = np.clip(((x_center + width / 2) * scale_x), 0, original_w)
            self.coord_buffer[:len(valid_boxes), 3] = np.clip(((y_center + height / 2) * scale_y), 0, original_h)
            
            # OPTIMIZATION: Filter small boxes before NMS
            box_areas = (self.coord_buffer[:len(valid_boxes), 2] - self.coord_buffer[:len(valid_boxes), 0]) * \
                       (self.coord_buffer[:len(valid_boxes), 3] - self.coord_buffer[:len(valid_boxes), 1])
            area_mask = box_areas >= 100  # Minimum 100 pixels
            
            if not np.any(area_mask):
                return []
            
            # Apply area filter
            valid_coords = self.coord_buffer[:len(valid_boxes)][area_mask]
            valid_confidences = confidences_sigmoid[area_mask]
            valid_class_ids = class_ids[area_mask]
            
            # SIMPLIFIED NMS: Fast overlap removal
            boxes_for_nms = valid_coords.astype(np.float32)
            
            indices = cv2.dnn.NMSBoxes(
                boxes_for_nms.tolist(),
                valid_confidences.tolist(),
                conf_threshold,
                0.4  # IoU threshold
            )
            
            if len(indices) == 0:
                return []
            
            top_indices = indices.flatten()
            
            # BUILD RESULTS: No masks for speed
            live_detections = []
            for idx, i in enumerate(top_indices):
                x1, y1, x2, y2 = valid_coords[i]
                
                detection = {
                    "id": f"live_{idx}",
                    "class_name": self.class_names.get(valid_class_ids[i], f"class_{valid_class_ids[i]}"),
                    "confidence": float(valid_confidences[i]),
                    "bbox": {
                        "x1": int(x1), "y1": int(y1), 
                        "x2": int(x2), "y2": int(y2)
                    },
                    "mask": None,  # No masks for live detection speed
                    "image_size": {"width": original_w, "height": original_h}
                }
                live_detections.append(detection)
            
            return live_detections
            
        except Exception as e:
            logger.error(f"Async fast live detection failed: {e}")
            return []
    
    def _run_inference_optimized(self, input_buffer):
        """Optimized inference with IOBinding for zero-copy"""
        try:
            # Create OrtValue from numpy array
            ort_value = ort.OrtValue.ortvalue_from_numpy(input_buffer)
            
            # Bind input
            self.io_binding.bind_input(self.input_name, ort_value)
            
            # Run inference
            self.session.run_with_iobinding(self.io_binding)
            
            # Get outputs
            outputs = self.io_binding.get_outputs()
            return [output.numpy() for output in outputs]
            
        except Exception as e:
            logger.error(f"IOBinding inference failed: {e}")
            # Fallback to regular inference
            return self.session.run(None, {self.input_name: input_buffer})

    def fast_live_detect(self, image, conf_threshold: float = 0.2, iou_threshold: float = 0.2, max_detections: int = 50) -> List[dict]:
        """
        ULTRA-FAST live detection optimized for real-time streaming
        
        Optimizations:
        - GPU acceleration (CUDA/CoreML)
        - Lower confidence threshold (0.2 default)
        - Lower IoU threshold (0.2 default)
        - No mask generation for speed
        - Limited max detections (50)
        - Quick NMS for overlap removal
        """
        try:
            # Use the main predict method with fast_mode=True
            detections = self.predict(
                image,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                fast_mode=True  # Enable fast mode for live detection
            )
            
            # Limit detections for performance
            if len(detections) > max_detections:
                detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)[:max_detections]
            
            return detections
            
        except Exception as e:
            logger.error(f"Fast live detection failed: {e}")
            return []

    def predict(self, image, conf_threshold: float = 0.25, iou_threshold: float = 0.3, fast_mode: bool = False) -> List[dict]:
        """Fast prediction optimized for live detection
        
        Args:
            image: PIL Image or numpy array
            conf_threshold: Confidence threshold (0.0-1.0)
            iou_threshold: IoU threshold for NMS (0.0-1.0)
            fast_mode: If True, skip mask generation for speed
        
        Returns:
            List of detection dictionaries
        """
        try:
            # Handle different input types efficiently
            if isinstance(image, Image.Image):
                img = np.array(image)
                logger.info(f"🖼️ PIL Image input: {img.shape}, min={img.min()}, max={img.max()}")
            elif isinstance(image, np.ndarray):
                img = image
                logger.info(f"📷 NumPy array input: {img.shape}, min={img.min()}, max={img.max()}")
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            original_h, original_w = img.shape[:2]
            logger.info(f"🎯 Processing {original_w}x{original_h} with confidence threshold: {conf_threshold}")
            
            # Resize to model input size
            img_resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
            img_norm = img_resized.astype(np.float32) / 255.0
            img_batch = np.transpose(img_norm, (2, 0, 1))[None, ...]
            
            # Run inference
            outputs = self.session.run(None, {self.input_name: img_batch})
            
            # Extract outputs
            detections = outputs[0][0].T
            boxes = detections[:, :4]
            class_scores = detections[:, 4:84]
            
            # Extract mask coefficients and prototypes (ALWAYS, even in fast mode)
            mask_coeffs_raw = detections[:, 84:]  # This might be 8316 coefficients
            mask_protos = outputs[1][0] if len(outputs) > 1 else None  # Shape: (32, 160, 160)
            
            # CRITICAL FIX: Limit mask coefficients to match prototype dimensions
            if mask_coeffs_raw.shape[1] > 32:
                logger.info(f"⚠️ Model outputs {mask_coeffs_raw.shape[1]} mask coeffs, limiting to 32")
                mask_coeffs = mask_coeffs_raw[:, :32]  # Take only first 32 coefficients
            else:
                mask_coeffs = mask_coeffs_raw
            
            # Apply confidence threshold (FIXED: compare sigmoid values, not raw logits)
            max_scores = np.max(class_scores, axis=1)
            confidences_sigmoid = 1 / (1 + np.exp(-max_scores))
            valid_indices = confidences_sigmoid > conf_threshold  # Compare sigmoid, not raw scores!
            
            # FAST MODE: Limit detections for live streaming
            if fast_mode and np.sum(valid_indices) > 10:
                # Keep only top 10 detections by confidence for speed
                valid_confidences = confidences_sigmoid[valid_indices]
                top_indices = np.argsort(valid_confidences)[-10:]  # Top 10
                temp_indices = np.where(valid_indices)[0]
                valid_indices = np.zeros_like(valid_indices, dtype=bool)
                valid_indices[temp_indices[top_indices]] = True
            
            logger.info(f"🔍 Raw detections: {len(max_scores)}, Max confidence: {np.max(confidences_sigmoid):.3f}, Above threshold: {np.sum(valid_indices)}")
            
            if np.sum(valid_indices) == 0:
                logger.info("❌ No detections above threshold")
                return []
            
            valid_boxes = boxes[valid_indices]
            valid_scores = class_scores[valid_indices]
            valid_mask_coeffs = mask_coeffs[valid_indices] if mask_coeffs is not None and mask_protos is not None else None
            
            # Get class IDs and confidences
            class_ids = np.argmax(valid_scores, axis=1)
            confidences = 1 / (1 + np.exp(-np.max(valid_scores, axis=1)))
            
            # Ultra-fast coordinate conversion and NMS
            valid_detections = []
            if len(valid_boxes) > 0:
                # Vectorized coordinate conversion (much faster)
                x_center, y_center, width, height = valid_boxes.T
                
                # Direct scaling without intermediate variables
                scale_x, scale_y = original_w / 640, original_h / 640
                
                x1 = np.clip(((x_center - width / 2) * scale_x).astype(int), 0, original_w)
                y1 = np.clip(((y_center - height / 2) * scale_y).astype(int), 0, original_h)
                x2 = np.clip(((x_center + width / 2) * scale_x).astype(int), 0, original_w)
                y2 = np.clip(((y_center + height / 2) * scale_y).astype(int), 0, original_h)
                
                # Apply proper NMS to remove overlapping detections
                # Convert to x1,y1,x2,y2 format for NMS
                boxes_for_nms = np.column_stack([x1, y1, x2, y2]).astype(np.float32)
                
                # Use OpenCV NMS with proper IoU threshold
                indices = cv2.dnn.NMSBoxes(
                    boxes_for_nms.tolist(),
                    confidences.tolist(),
                    conf_threshold,
                    iou_threshold  # Use the parameter instead of hardcoded value
                )
                
                # Get valid indices after NMS
                if len(indices) > 0:
                    top_indices = indices.flatten()
                else:
                    top_indices = []
                
                for idx, i in enumerate(top_indices):
                    # Double-check confidence threshold after NMS
                    if confidences[i] > conf_threshold:
                        # Additional filtering: remove very small boxes
                        box_width = x2[i] - x1[i]
                        box_height = y2[i] - y1[i]
                        box_area = box_width * box_height
                        
                        # Skip very small detections (likely noise)
                        if box_area < 100:  # Minimum 100 pixels area
                            continue
                        # Generate mask if available (NOW WORKS IN FAST MODE TOO!)
                        mask_points = None
                        if valid_mask_coeffs is not None and mask_protos is not None:
                            try:
                                # Generate mask from coefficients and prototypes
                                mask = np.dot(valid_mask_coeffs[i], mask_protos.reshape(32, -1))
                                mask = mask.reshape(160, 160)
                                mask = 1 / (1 + np.exp(-mask))  # Sigmoid activation
                                
                                # Resize mask to original image size
                                mask_resized = cv2.resize(mask, (original_w, original_h))
                                
                                # Apply mask only within bounding box area for better accuracy
                                mask_roi = np.zeros_like(mask_resized)
                                mask_roi[y1[i]:y2[i], x1[i]:x2[i]] = mask_resized[y1[i]:y2[i], x1[i]:x2[i]]
                                
                                # Convert to contour points with better threshold
                                mask_binary = (mask_roi > 0.5).astype(np.uint8) * 255
                                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if contours:
                                    # Filter contours by area and select the one closest to bbox center
                                    bbox_center_x = (x1[i] + x2[i]) // 2
                                    bbox_center_y = (y1[i] + y2[i]) // 2
                                    
                                    valid_contours = [c for c in contours if cv2.contourArea(c) > box_area * 0.1]
                                    if valid_contours:
                                        # Select contour closest to bbox center
                                        best_contour = min(valid_contours, key=lambda c: 
                                            abs(cv2.moments(c)['m10']/cv2.moments(c)['m00'] - bbox_center_x) + 
                                            abs(cv2.moments(c)['m01']/cv2.moments(c)['m00'] - bbox_center_y)
                                            if cv2.moments(c)['m00'] > 0 else float('inf'))
                                        mask_points = best_contour.reshape(-1, 2).tolist()
                            except Exception as e:
                                logger.warning(f"Mask generation failed: {e}")
                        
                        detection = {
                            "id": f"det_{idx}",
                            "class_name": self.class_names.get(class_ids[i], f"class_{class_ids[i]}"),
                            "confidence": float(confidences[i]),
                            "bbox": {
                                "x1": int(x1[i]), "y1": int(y1[i]), 
                                "x2": int(x2[i]), "y2": int(y2[i])
                            },
                            "mask": mask_points,
                            "image_size": {"width": original_w, "height": original_h}
                        }
                        valid_detections.append(detection)
            
            logger.info(f"✅ Returning {len(valid_detections)} detections")
            if valid_detections:
                for det in valid_detections:
                    logger.info(f"   📦 {det['class_name']}: {det['confidence']:.3f}")
            return valid_detections
            
        except Exception as e:
            logger.error(f"Fast ONNX prediction failed: {e}")
            return []