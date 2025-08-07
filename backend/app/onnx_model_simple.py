import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple
import logging
import asyncio
import time

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
    
    async def fast_live_detect_async(self, image, conf_threshold: float = 0.6, iou_threshold: float = 0.3, max_detections: int = 50) -> List[dict]:
        """
        ASYNC live detection + segmentation masks
        - GPU + IOBinding + pre-alloc buffers
        - Only generates masks, no bounding boxes for better performance
        """
        try:
            # 1) load & prep
            if isinstance(image, Image.Image):
                img = np.array(image)
            elif isinstance(image, np.ndarray):
                img = image
            else:
                raise ValueError(f"Unsupported type {type(image)}")
            h0, w0 = img.shape[:2]

            # 2) resize & normalize into our buffer
            small = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
            norm = small.astype(np.float32) / 255.0
            np.copyto(self.input_buffer[0], np.transpose(norm, (2,0,1)))

            # 3) GPU inference
            outputs = self._run_inference_optimized(self.input_buffer)

            # 4) Process outputs
            preds = outputs[0][0].T  # (anchors,116)
            class_scores = preds[:, 4:84]  # (anchors, 80)
            mask_coeffs_raw = preds[:, 84:]  # (anchors, 32+)
            proto = outputs[1][0] if len(outputs)>1 else None  # (32,160,160)

            # 5) Calculate confidences and filter
            class_scores_sigmoid = 1 / (1 + np.exp(-class_scores))
            confidences = np.max(class_scores_sigmoid, axis=1)
            class_ids = np.argmax(class_scores_sigmoid, axis=1)

            # Log confidence distribution
            logger.debug(f"Confidence range: {np.min(confidences):.3f} - {np.max(confidences):.3f}")
            logger.debug(f"Detections above threshold {conf_threshold}: {np.sum(confidences > conf_threshold)}")

            # Filter by confidence
            valid_indices = confidences > conf_threshold
            if not valid_indices.any():
                return []

            # Get valid detections
            valid_scores = confidences[valid_indices]
            valid_class_ids = class_ids[valid_indices]
            valid_boxes = preds[valid_indices, :4]  # Get bounding boxes for NMS

            # Convert YOLO format to x1,y1,x2,y2 for NMS
            x_center, y_center, width, height = valid_boxes.T
            scale_x, scale_y = w0 / 640, h0 / 640

            x1 = ((x_center - width / 2) * scale_x)
            y1 = ((y_center - height / 2) * scale_y)
            x2 = ((x_center + width / 2) * scale_x)
            y2 = ((y_center + height / 2) * scale_y)

            boxes_for_nms = np.column_stack([x1, y1, x2, y2])

            # Apply NMS to remove duplicates based on IoU threshold
            nms_indices = cv2.dnn.NMSBoxes(
                boxes_for_nms.tolist(),
                valid_scores.tolist(),
                score_threshold=0.0001,  # Very low since we already filtered
                nms_threshold=iou_threshold
            )

            if len(nms_indices) == 0:
                return []

            nms_indices = nms_indices.flatten()

            # Limit mask coefficients to match prototype dimensions
            if mask_coeffs_raw.shape[1] > 32:
                mask_coeffs = mask_coeffs_raw[:, :32]
            else:
                mask_coeffs = mask_coeffs_raw
            valid_mask_coeffs = mask_coeffs[valid_indices]

            # Sort NMS results by confidence and limit
            sorted_nms = sorted(nms_indices, key=lambda i: valid_scores[i], reverse=True)[:max_detections]

            # 6) Generate results
            results = []
            for rank, i in enumerate(sorted_nms):
                det = {
                    "id": f"det_{rank}",
                    "class_name": self.class_names[valid_class_ids[i]],
                    "confidence": float(valid_scores[i]),
                    "bbox": {
                        "x1": int(x1[i]),
                        "y1": int(y1[i]),
                        "x2": int(x2[i]),
                        "y2": int(y2[i])
                    },
                    "mask": None,
                    "image_size": {"width": w0, "height": h0}
                }

                # 7) Generate mask
                if proto is not None:
                    try:
                        # Generate full mask
                        m = valid_mask_coeffs[i] @ proto.reshape(32, -1)
                        m = m.reshape(160, 160)
                        m = 1/(1 + np.exp(-m))  # sigmoid
                        m = cv2.resize(m, (w0, h0)) > 0.5  # resize to original size

                        # Find contours directly on the full mask
                        cnts, _ = cv2.findContours((m * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts:
                            # Get largest contour
                            largest = max(cnts, key=cv2.contourArea)
                            det["mask"] = largest.reshape(-1, 2).tolist()

                    except Exception as e:
                        logger.warning(f"Mask generation failed: {e}")

                results.append(det)

            return results

        except Exception as e:
            logger.error(f"live+mask failed: {e}")
            return []

    def _run_inference_optimized(self, input_buffer):
        """Optimized inference with IOBinding for zero-copy"""
        try:
            inference_start = time.time()
            
            # Clear any existing bindings
            self.io_binding.clear_binding_inputs()
            self.io_binding.clear_binding_outputs()
            
            # Bind input tensor directly from numpy array
            self.io_binding.bind_cpu_input(
                self.input_name,
                input_buffer
            )
            
            # Bind output tensors
            for output in self.session.get_outputs():
                self.io_binding.bind_output(output.name)
            
            # Run inference
            self.session.run_with_iobinding(self.io_binding)
            
            # Get outputs
            outputs = [output.numpy() for output in self.io_binding.get_outputs()]
            
            inference_time = time.time() - inference_start
            logger.info(f"⚡ Pure ONNX inference time (optimized): {inference_time:.3f}s")
            
            return outputs
            
        except Exception as e:
            logger.error(f"IOBinding inference failed: {e}")
            # Fallback to regular inference
            inference_start = time.time()
            outputs = self.session.run(None, {self.input_name: input_buffer})
            inference_time = time.time() - inference_start
            logger.info(f"⚡ Pure ONNX inference time (fallback): {inference_time:.3f}s")
            return outputs

    def fast_live_detect(self, image, conf_threshold: float = 0.2, iou_threshold: float = 0.2, max_detections: int = 50) -> List[dict]:
        """
        ULTRA-FAST live detection optimized for real-time streaming
        
        Optimizations:
        - GPU acceleration (CUDA/CoreML)
        - Pre-allocated buffers
        - Efficient mask generation
        - Limited max detections
        """
        try:
            # Use the main predict method with fast_mode=True
            detections = self.predict(
                image,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                fast_mode=True  # Enable fast mode but still generate masks
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
                logger.info(f"�� NumPy array input: {img.shape}, min={img.min()}, max={img.max()}")
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            original_h, original_w = img.shape[:2]
            logger.info(f"🎯 Processing {original_w}x{original_h} with confidence threshold: {conf_threshold}")
            
            # Resize to model input size
            img_resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
            img_norm = img_resized.astype(np.float32) / 255.0
            img_batch = np.transpose(img_norm, (2, 0, 1))[None, ...]
            
            # Run inference with timing
            inference_start = time.time()
            outputs = self.session.run(None, {self.input_name: img_batch})
            inference_time = time.time() - inference_start
            logger.info(f"⚡ Pure ONNX inference time (predict): {inference_time:.3f}s")
            
            # Extract outputs
            detections = outputs[0][0].T  # Shape: (8400, 116)
            boxes = detections[:, :4]  # (8400, 4)
            class_scores = detections[:, 4:84]  # (8400, 80)
            mask_coeffs_raw = detections[:, 84:]  # (8400, 32+)
            mask_protos = outputs[1][0] if len(outputs) > 1 else None  # (32, 160, 160)
            
            # Calculate confidences and class IDs
            class_scores_sigmoid = 1 / (1 + np.exp(-class_scores))
            confidences = np.max(class_scores_sigmoid, axis=1)
            class_ids = np.argmax(class_scores_sigmoid, axis=1)
            
            # Log detection stats
            logger.info(f"🔍 Raw detections: {len(confidences)}, Max confidence: {np.max(confidences):.3f}, Above threshold: {np.sum(confidences > conf_threshold)}")
            
            # Filter by confidence threshold
            valid_indices = confidences > conf_threshold
            valid_boxes = boxes[valid_indices]
            valid_confidences = confidences[valid_indices]
            valid_class_ids = class_ids[valid_indices]
            
            # Log valid detections
            if np.sum(valid_indices) > 0:
                logger.info("Valid detections:")
                for i, (cls_id, conf) in enumerate(zip(valid_class_ids[:5], valid_confidences[:5])):
                    logger.info(f"  {i}: class={self.class_names.get(cls_id, f'class_{cls_id}')}, score={conf:.3f}")
            
            # Handle mask coefficients
            if mask_coeffs_raw is not None and mask_protos is not None:
                if mask_coeffs_raw.shape[1] > 32:
                    mask_coeffs = mask_coeffs_raw[:, :32]  # Take only first 32 coefficients
                else:
                    mask_coeffs = mask_coeffs_raw
                valid_mask_coeffs = mask_coeffs[valid_indices]
            else:
                valid_mask_coeffs = None
            
            # Convert coordinates and apply NMS
            valid_detections = []
            if len(valid_boxes) > 0:
                # Convert coordinates
                x_center, y_center, width, height = valid_boxes.T
                scale_x, scale_y = original_w / 640, original_h / 640
                
                x1 = np.clip(((x_center - width / 2) * scale_x).astype(int), 0, original_w)
                y1 = np.clip(((y_center - height / 2) * scale_y).astype(int), 0, original_h)
                x2 = np.clip(((x_center + width / 2) * scale_x).astype(int), 0, original_w)
                y2 = np.clip(((y_center + height / 2) * scale_y).astype(int), 0, original_h)
                
                # Prepare boxes for NMS
                boxes_for_nms = np.column_stack([x1, y1, x2, y2])
                
                # Apply NMS
                indices = cv2.dnn.NMSBoxes(
                    boxes_for_nms.tolist(),
                    valid_confidences.tolist(),
                    score_threshold=0.0001,  # Use very low score threshold since we already filtered
                    nms_threshold=iou_threshold  # This is where IoU filtering happens
                )
                
                if len(indices) > 0:
                    indices = indices.flatten()
                    
                    # Sort by confidence and limit detections
                    sorted_indices = sorted(indices, key=lambda i: valid_confidences[i], reverse=True)
                    
                    # Create detection objects
                    for i in sorted_indices:
                        # Skip if we have too many detections of this class
                        class_name = self.class_names.get(valid_class_ids[i], f"class_{valid_class_ids[i]}")
                        class_count = sum(1 for d in valid_detections if d["class_name"] == class_name)
                        if class_count >= 3:  # Limit detections per class
                            continue
                            
                        detection = {
                            "id": f"det_{i}",
                            "class_name": class_name,
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
                                # Generate mask
                                mask = np.dot(valid_mask_coeffs[i], mask_protos.reshape(32, -1))
                                mask = mask.reshape(160, 160)
                                mask = 1 / (1 + np.exp(-mask))  # Sigmoid
                                
                                # Resize mask to original image size
                                mask_resized = cv2.resize(mask, (original_w, original_h))
                                
                                # Get mask for bbox region
                                mask_roi = mask_resized[y1[i]:y2[i], x1[i]:x2[i]] > 0.5
                                
                                if mask_roi.size > 0:
                                    # Find contours
                                    cnts, _ = cv2.findContours(
                                        (mask_roi * 255).astype(np.uint8),
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE
                                    )
                                    
                                    if cnts:
                                        # Get largest contour
                                        largest = max(cnts, key=cv2.contourArea)
                                        # Offset points by bbox coordinates
                                        pts = (largest + [x1[i], y1[i]]).reshape(-1, 2).tolist()
                                        detection["mask"] = pts
                            except Exception as e:
                                logger.warning(f"Mask generation failed: {e}")
                                logger.info(f"Proto shape: {mask_protos.shape}")
                                logger.info(f"Valid mask coeffs shape: {valid_mask_coeffs[i].shape}")
                        
                        valid_detections.append(detection)
            
            return valid_detections
            
        except Exception as e:
            logger.error(f"Fast ONNX prediction failed: {e}")
            return []