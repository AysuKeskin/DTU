from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import base64
from typing import List, Optional
import time
from pydantic import BaseModel
import logging
import sys
import os
import torch
import asyncio
import json
from threading import Thread
import queue
import tempfile

# Import our new video processor
from .video_processor import VideoProcessor
from .onnx_model_simple import SimpleONNXModel
from PIL import Image
import io

# =============================================================================
# PERFORMANCE OPTIMIZATION NOTES
# =============================================================================
# 🚀 OPTIMIZATION: Let YOLO handle resizing internally
# - Removed manual image resizing in backend
# - YOLO automatically resizes to imgsz=640 for optimal performance
# - BBOXES: Returned in original image space (no scaling needed)
# - MASKS: Returned in inference resolution, need cv2.resize() to original dimensions
# - This eliminates coordinate scaling issues and improves speed
# - Masks now positioned correctly with bounding boxes!
#
# 🚀 WEBSOCKET PERFORMANCE OPTIMIZATIONS:
# - Reduced logging frequency: Only log every 30 frames instead of every frame
# - FPS monitoring: Log performance every 5 seconds
# - Error logging: Only log every 10th error to avoid spam
# - Removed excessive print statements that were causing slowdown
# - This should provide consistent 30+ FPS for real-time detection
# =============================================================================

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# =============================================================================
# FASTAPI APP SETUP
# =============================================================================
app = FastAPI(title="DTU Aqua Vision API")

# CORS configuration for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://localhost:8001",
        "http://localhost:8002",
        "http://localhost:8003"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DATA MODELS (Pydantic schemas)
# =============================================================================
class Detection(BaseModel):
    """Single detection result with bounding box, mask, and metadata"""
    id: str
    class_name: str
    confidence: float
    bbox: dict
    measurements: Optional[dict] = None
    mask: Optional[List[List[float]]] = None  # Segmentation mask points
    image_size: Optional[dict] = None  # Original image dimensions

class DetectionResponse(BaseModel):
    """Response for single image detection"""
    detections: List[Detection]
    processed_image_url: Optional[str] = None
    processing_time: float
    model_version: str

class VideoDetectionResponse(BaseModel):
    """Response for video detection with multiple frames"""
    total_frames: int
    processed_frames: int
    detections_per_frame: List[List[Detection]]
    processing_time: float
    model_version: str
    video_info: dict

# =============================================================================
# GLOBAL VARIABLES & MODEL MANAGEMENT
# =============================================================================
# Global model variable
current_model_name = None
onnx_model = None

# =============================================================================
# STREAMING INFRASTRUCTURE (for real-time camera detection)
# =============================================================================
frame_queue = queue.Queue(maxsize=2)  # Small queue to avoid lag
detection_results = {}
websocket_connections = set()
streaming_active = False

# =============================================================================
# DEVICE DETECTION (ONNX Runtime)
# =============================================================================
# Detect best available device for ONNX Runtime
import onnxruntime as ort
providers = ort.get_available_providers()

if 'CUDAExecutionProvider' in providers:
    device = 'cuda'
    device_name = 'CUDA GPU'
elif 'CoreMLExecutionProvider' in providers:
    device = 'coreml'
    device_name = 'Apple Silicon GPU (CoreML)'
else:
    device = 'cpu'
    device_name = 'CPU'

logger.info(f"Using device: {device} ({device_name})")
logger.info(f"Available ONNX providers: {providers}")

# Available model options
AVAILABLE_MODELS = {
    "yolov8n-seg.onnx": {"name": "YOLOv8n ONNX", "description": "Properly converted ONNX model", "size": "13 MB"}
}

def load_onnx_model(model_key: str):
    """Load a specific ONNX model"""
    global onnx_model, current_model_name
    
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_key} not available. Available models: {list(AVAILABLE_MODELS.keys())}")
    
    try:
        logger.info(f"Loading ONNX model: {model_key}")
        model_path = os.path.join(os.path.dirname(__file__), '..', model_key)
        onnx_model = SimpleONNXModel(model_path)
        current_model_name = model_key
        logger.info(f"✅ ONNX model '{model_key}' loaded successfully on {device}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load ONNX model {model_key}: {e}")
        return False

# Load default ONNX model at startup with proper warmup
try:
    logger.info("🚀 Starting backend initialization...")
    
    # Load the ONNX model
    default_model = "yolov8n-seg.onnx"
    
    if load_onnx_model(default_model):
        logger.info(f"✅ Default ONNX model loaded: {default_model}")
        
        # IMPORTANT: Comprehensive model warmup for real-world performance
        logger.info("🔥 Starting comprehensive ONNX model warmup...")
        try:
            import numpy as np
            from PIL import Image
            
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
                    detections = onnx_model.predict(dummy_image, conf_threshold=0.25)
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
                detections = onnx_model.predict(test_image, conf_threshold=conf)
                test_time = time.time() - test_start
                logger.info(f"   • Confidence {conf}: {test_time:.3f}s")
            
            logger.info(f"🔥 Comprehensive ONNX warmup completed in {warmup_time:.2f}s")
            logger.info(f"✅ ONNX model is fully optimized and ready for production use!")
        except Exception as warmup_error:
            logger.warning(f"⚠️ ONNX model warmup failed: {warmup_error}, but model is loaded")
        
    else:
        raise Exception("Failed to load ONNX model")
    
    logger.info("✅ Backend initialization complete - ready to accept requests!")
        
except Exception as e:
    logger.error(f"❌ Failed to initialize backend: {e}")
    onnx_model = None


# =============================================================================
# IMAGE DETECTION ENDPOINTS
# =============================================================================

@app.post("/detect", response_model=DetectionResponse)
async def detect_fish(
    file: UploadFile = File(...),
    confidence: float = 0.6,
    iou_threshold: float = 0.7
):
    """
    MAIN IMAGE DETECTION ENDPOINT
    =============================
    Handles single image upload and detection for:
    - Image Mode: Multiple images processed individually
    - Single image uploads
    - Returns detection results with bounding boxes and masks
    """
    try:
        start_time = time.time()
        step_times = {}
        
        logger.info(f"🚀 Starting detection for: {file.filename} with confidence={confidence}, IoU={iou_threshold}")
        
        # Check if model is loaded
        if onnx_model is None:
            logger.error("❌ ONNX model not loaded!")
            raise HTTPException(status_code=500, detail="ONNX model not loaded")
        
        logger.info(f"✅ ONNX model loaded: {current_model_name} on {device}")
        
        # Step 1: Read and decode file
        step_start = time.time()
        contents = await file.read()
        logger.info(f"📥 File read: {len(contents)} bytes")
        
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_width, original_height = image.size
        step_times['file_processing'] = time.time() - step_start
        logger.info(f"📂 File processing: {step_times['file_processing']:.3f}s - Size: {image.size}")
        
        # Check for reasonable image size
        if original_width > 2000 or original_height > 2000:
            logger.warning(f"⚠️ Large image detected: {original_width}x{original_height} - this may be slow")
        
        # Step 2: ONNX inference with timeout protection
        step_start = time.time()
        logger.info(f"🤖 Starting ONNX inference with confidence={confidence}")
        try:
            # Run ONNX inference
            detections = onnx_model.predict(image, conf_threshold=confidence, iou_threshold=iou_threshold)
            step_times['onnx_inference'] = time.time() - step_start
            logger.info(f"🤖 ONNX inference completed: {step_times['onnx_inference']:.3f}s")
            
            # Check if inference took too long
            if step_times['onnx_inference'] > 5.0:
                logger.warning(f"⚠️ Slow ONNX inference: {step_times['onnx_inference']:.3f}s")
                
        except Exception as e:
            inference_time = time.time() - step_start
            logger.error(f"❌ ONNX inference failed after {inference_time:.3f}s: {e}")
            # Return empty response instead of crashing
            return DetectionResponse(
                detections=[],
                processed_image_url="",
                processing_time=inference_time,
                model_version=f"Error: {str(e)}"
            )
        
        # Step 3: Process ONNX results
        step_start = time.time()
        
        # ONNX model already returns detections in the correct format
        logger.info(f"📊 Found {len(detections)} detections from ONNX model")
        
        # Convert to Detection objects
        detection_objects = []
        for i, detection in enumerate(detections):
            detection_obj = Detection(
                id=detection["id"],
                class_name=detection["class_name"],
                confidence=detection["confidence"],
                bbox=detection["bbox"],
                        measurements={
                    "length": (detection["bbox"]["x2"] - detection["bbox"]["x1"]) * 0.1,
                    "width": (detection["bbox"]["y2"] - detection["bbox"]["y1"]) * 0.1,
                    "area": (detection["bbox"]["x2"] - detection["bbox"]["x1"]) * (detection["bbox"]["y2"] - detection["bbox"]["y1"]) * 0.01
                },
                mask=detection["mask"],
                image_size=detection["image_size"]
            )
            detection_objects.append(detection_obj)
            
        step_times['result_processing'] = time.time() - step_start
        logger.info(f"📝 Result processing: {step_times['result_processing']:.3f}s")
        
        # Step 4: Draw annotations and encode image
        step_start = time.time()
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Draw boxes
        for detection in detection_objects:
            bbox = detection.bbox
            cv2.rectangle(img_bgr, (bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y2"]), (0, 255, 0), 2)
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(img_bgr, label, (bbox["x1"], bbox["y1"]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        step_times['image_encoding'] = time.time() - step_start
        logger.info(f"🖼️  Image encoding: {step_times['image_encoding']:.3f}s")
        
        processing_time = time.time() - start_time
        model_info = f"{AVAILABLE_MODELS.get(current_model_name, {}).get('name', current_model_name)} on {device}"
        
        # Summary timing log
        logger.info(f"⚡ TOTAL: {processing_time:.3f}s | File: {step_times['file_processing']:.3f}s | ONNX: {step_times['onnx_inference']:.3f}s | Processing: {step_times['result_processing']:.3f}s | Encoding: {step_times['image_encoding']:.3f}s | Objects: {len(detection_objects)}")
        
        return DetectionResponse(
            detections=detection_objects,
            processed_image_url=f"data:image/jpeg;base64,{image_base64}",
            processing_time=processing_time,
            model_version=model_info
        )
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/test-detect", response_model=DetectionResponse)
async def test_detect(
    file: UploadFile = File(...),
    confidence: float = 0.5
):
    """Test detection endpoint that returns fake results to verify system works"""
    try:
        start_time = time.time()
        
        # Read file quickly
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        width, height = image.size
        
        # Create fake detections for testing
        fake_detections = [
            Detection(
                id="test_1",
                class_name="person",
                confidence=0.85,
                bbox={"x1": 100, "y1": 100, "x2": 300, "y2": 400},
                measurements={"length": 20.0, "width": 10.0, "area": 200.0},
                mask=[[120, 120], [280, 120], [280, 380], [120, 380]],  # Simple rectangle
                image_size={"width": width, "height": height}
            ),
            Detection(
                id="test_2", 
                class_name="cell phone",
                confidence=0.92,
                bbox={"x1": 400, "y1": 200, "x2": 500, "y2": 350},
                measurements={"length": 10.0, "width": 5.0, "area": 50.0},
                mask=[[410, 210], [490, 210], [490, 340], [410, 340]],  # Simple rectangle
                image_size={"width": width, "height": height}
            )
        ]
        
        # Simple image with rectangles
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Draw test boxes
        cv2.rectangle(img_bgr, (100, 100), (300, 400), (0, 255, 0), 2)
        cv2.putText(img_bgr, "person: 0.85", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(img_bgr, (400, 200), (500, 350), (0, 255, 0), 2) 
        cv2.putText(img_bgr, "phone: 0.92", (400, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Encode image
        _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Test detection completed in {processing_time:.3f}s")
        
        return DetectionResponse(
            detections=fake_detections,
            processed_image_url=f"data:image/jpeg;base64,{image_base64}",
            processing_time=processing_time,
            model_version="Test Mode - Fake Detections"
        )
        
    except Exception as e:
        logger.error(f"❌ Test detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Test detection failed: {str(e)}")

# =============================================================================
# UTILITY ENDPOINTS (Model management, health checks, etc.)
# =============================================================================

@app.get("/models")
async def get_available_models():
    """
    MODEL LIST ENDPOINT
    ===================
    Returns available ONNX models for:
    - Frontend model selection dropdown
    - Model information display
    """
    return {
        "current_model": current_model_name,
        "device": device,
        "device_name": device_name,
        "available_models": AVAILABLE_MODELS
    }

@app.post("/models/{model_key}")
async def change_model(model_key: str):
    """
    MODEL CHANGE ENDPOINT
    ====================
    Changes the active ONNX model for:
    - Switching between different ONNX models
    - Model performance optimization
    - Different detection capabilities
    """
    try:
        logger.info(f"🔄 ONNX model change requested: {model_key}")
        
        if model_key == current_model_name:
            logger.info(f"✅ ONNX model {model_key} already loaded")
            return {
                "success": True,
                "message": f"ONNX model {model_key} already active",
                "current_model": current_model_name,
                "model_info": AVAILABLE_MODELS[model_key],
                "loading_time": 0.0
            }
        
        start_time = time.time()
        if load_onnx_model(model_key):
            loading_time = time.time() - start_time
            logger.info(f"✅ ONNX model change completed in {loading_time:.2f}s")
            return {
                "success": True,
                "message": f"ONNX model changed to {model_key}",
                "current_model": current_model_name,
                "model_info": AVAILABLE_MODELS[model_key],
                "loading_time": loading_time
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to load ONNX model {model_key}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error changing ONNX model: {str(e)}")

@app.get("/model-status")
async def get_model_status():
    """
    MODEL STATUS ENDPOINT
    ====================
    Returns current ONNX model loading status for:
    - Frontend status indicators
    - Model readiness checks
    """
    return {
        "model_loaded": onnx_model is not None,
        "current_model": current_model_name,
        "device": device,
        "device_name": device_name,
        "ready_for_detection": onnx_model is not None,
        "model_info": AVAILABLE_MODELS.get(current_model_name, {}) if current_model_name else {}
    }

@app.get("/health")
async def health_check():
    """
    HEALTH CHECK ENDPOINT
    ====================
    Comprehensive API health check for:
    - Backend readiness verification
    - Model inference capability testing
    - Frontend connection status
    """
    model_ready = onnx_model is not None
    
    # Test if model can actually run inference
    inference_ready = False
    if model_ready:
        try:
            # Quick readiness test
            test_start = time.time()
            import numpy as np
            from PIL import Image
            test_array = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
            test_image = Image.fromarray(test_array)
            _ = onnx_model.predict(test_image, conf_threshold=0.25)
            test_time = time.time() - test_start
            inference_ready = True
            inference_test_time = test_time
        except Exception as e:
            logger.warning(f"Health check inference test failed: {e}")
            inference_test_time = None
    else:
        inference_test_time = None
    
    status = "ready" if (model_ready and inference_ready) else "starting"
    
    return {
        "status": status,
        "model_type": "ONNX YOLO11n Segmentation", 
        "current_model": current_model_name,
        "device": device,
        "device_name": device_name,
        "model_loaded": model_ready,
        "inference_ready": inference_ready,
        "inference_test_time": inference_test_time,
        "ready_for_requests": model_ready and inference_ready
    } 

# =============================================================================
# STREAMING ENDPOINTS (Real-time camera detection)
# =============================================================================

@app.get("/stream")
async def video_stream():
    """
    MJPEG STREAMING ENDPOINT
    ========================
    Provides MJPEG video stream for:
    - Live Camera Mode: Real-time video streaming
    - Used by frontend to display camera feed
    - Returns multipart/x-mixed-replace stream
    """
    def generate_frames():
        while streaming_active:
            try:
                # Get frame from queue (blocking with timeout)
                if not frame_queue.empty():
                    frame_data = frame_queue.get(timeout=0.1)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                else:
                    time.sleep(0.01)  # Small delay to prevent busy waiting
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                break
                
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WEBSOCKET ENDPOINT FOR REAL-TIME DETECTION
    ==========================================
    Handles real-time detection via WebSocket for:
    - Live Camera Mode: Real-time object detection
    - Receives frames from frontend camera
    - Returns detection results immediately
    - OPTIMIZED FOR MAXIMUM SPEED: Reduced logging, FPS monitoring
    """
    await websocket.accept()
    
    # Parse query parameters from WebSocket URL
    query_params = websocket.query_params
    confidence = float(query_params.get('confidence', 0.6))
    iou_threshold = float(query_params.get('iou_threshold', 0.7))
    
    logger.info(f"🔌 WebSocket connected for real-time detection with confidence: {confidence}, IoU: {iou_threshold}")
    
    # PERFORMANCE MONITORING: Initialize counters
    websocket_endpoint.frame_count = 0
    websocket_endpoint.error_count = 0
    websocket_endpoint.start_time = time.time()
    websocket_endpoint.last_fps_log = time.time()
    
    try:
        while True:
            # Receive frame data from frontend
            data = await websocket.receive_bytes()
            # Removed frame logging for maximum speed
            
            try:
                # Decode the frame
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Convert frame for ONNX (BGR → RGB conversion handled in model)
                    # Note: frame is in BGR format from OpenCV, model will convert it properly
                    
                    # Run FAST live detection optimized for streaming (non-blocking)
                    # Use the async method directly
                    # Convert to RGB exactly as in your image endpoint
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    detections = await onnx_model.fast_live_detect_async(
                        rgb_frame, 
                        conf_threshold=confidence,
                        iou_threshold=iou_threshold,
                        max_detections=5  # Limit max detections to prevent duplicates
                    )

                    
                    # PERFORMANCE OPTIMIZATION: Reduced logging frequency
                    # Only log every 30 frames to avoid performance impact
                    frame_count = getattr(websocket_endpoint, 'frame_count', 0) + 1
                    websocket_endpoint.frame_count = frame_count
                    
                    if len(detections) > 0 and frame_count % 30 == 0:
                        logger.info(f"📦 Processing frame {frame_count}: {len(detections)} detections")
                    
                    # Convert detections to streaming format (masks only)
                    streaming_detections = []
                    for detection in detections:
                        # Only include necessary data for mask drawing
                        streaming_detection = {
                            "class_name": detection["class_name"],
                            "confidence": detection["confidence"],
                            "mask": detection["mask"],
                            "image_size": {"width": frame.shape[1], "height": frame.shape[0]}
                        }
                        streaming_detections.append(streaming_detection)
                    
                    # Debug: Log mask information
                    mask_count = sum(1 for det in streaming_detections if det["mask"] is not None)
                    if len(streaming_detections) > 0:
                        logger.info(f"🎭 WebSocket sending {len(streaming_detections)} detections, {mask_count} with masks")
                        for i, det in enumerate(streaming_detections[:3]):  # Log first 3
                            mask_info = f"mask: {len(det['mask'])} points" if det['mask'] else "no mask"
                            logger.info(f"  Detection {i}: {det['class_name']} ({det['confidence']:.2f}) - {mask_info}")
                    
                    # Send detection results back immediately
                    response_data = {
                        "detections": streaming_detections,
                        "timestamp": time.time(),
                        "status": "success"
                    }
                    
                    await websocket.send_text(json.dumps(response_data))
                    
                    # PERFORMANCE MONITORING: Log FPS every 5 seconds
                    current_time = time.time()
                    if current_time - websocket_endpoint.last_fps_log >= 5.0:
                        elapsed_time = current_time - websocket_endpoint.start_time
                        fps = websocket_endpoint.frame_count / elapsed_time if elapsed_time > 0 else 0
                        logger.info(f"🚀 WebSocket Performance: {fps:.1f} FPS, {websocket_endpoint.frame_count} frames processed")
                        websocket_endpoint.last_fps_log = current_time
                
            except Exception as e:
                logger.error(f"Detection processing error: {e}")
                # Send error response
                error_response = {
                    "detections": [],
                    "error": str(e),
                    "status": "error"
                }
                await websocket.send_text(json.dumps(error_response))
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("🔌 WebSocket disconnected")

@app.post("/start-stream")
async def start_stream():
    """Start the video streaming process"""
    global streaming_active
    
    if streaming_active:
        return {"status": "already_streaming", "stream_url": "/stream"}
    
    streaming_active = True
    logger.info("🚀 Starting video stream...")
    
    try:
        # Start the detection thread
        detection_thread = Thread(target=detection_worker, daemon=True)
        detection_thread.start()
        
        # Give it a moment to initialize
        await asyncio.sleep(0.5)
        
        logger.info("✅ Streaming started successfully")
        return {
            "status": "streaming_started", 
            "stream_url": "/stream",
            "websocket_url": "/ws",
            "message": "Camera detection worker started"
        }
    except Exception as e:
        streaming_active = False
        logger.error(f"Failed to start streaming: {e}")
        return {
            "status": "error",
            "message": f"Failed to start streaming: {str(e)}"
        }

@app.post("/stop-stream") 
async def stop_stream():
    """Stop the video streaming process"""
    global streaming_active
    streaming_active = False
    
    # Clear queues
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            break
            
    return {"status": "streaming_stopped"} 

@app.post("/send-frame")
async def send_frame(frame: UploadFile = File(...)):
    """Receive a frame from the frontend for processing"""
    try:
        if not streaming_active:
            logger.warning("⚠️ Frame received but streaming not active")
            return {"status": "error", "message": "Streaming not active"}
        
        # Read frame data
        frame_data = await frame.read()
        logger.info(f"📸 Received frame: {len(frame_data)} bytes")
        
        # Add to processing queue (non-blocking)
        if frame_queue.full():
            try:
                frame_queue.get_nowait()  # Remove old frame
                logger.debug("🔄 Replaced old frame in queue")
            except queue.Empty:
                pass
        
        frame_queue.put(frame_data)
        logger.info(f"✅ Frame added to queue, queue size: {frame_queue.qsize()}")
        
        return {"status": "success", "message": "Frame received"}
    except Exception as e:
        logger.error(f"❌ Frame receive error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/force-detect")
async def force_detect():
    """Force YOLO detection on the next available frame"""
    global detection_results
    
    try:
        if not streaming_active:
            return {"status": "error", "message": "Streaming not active"}
        
        if frame_queue.empty():
            return {"status": "error", "message": "No frames available"}
        
        # Get the latest frame
        frame_data = frame_queue.get_nowait()
        
        # Decode the frame
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"status": "error", "message": "Failed to decode frame"}
        
        logger.info(f"🔍 Force detection on frame, shape: {frame.shape}")
        
        # Convert frame for YOLO
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Run YOLO inference
        results = yolo_model(pil_image, conf=0.3, verbose=False)
        
        # Process detections
        detections = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                logger.info(f"📦 Found {len(r.boxes)} detections")
                for i, (box, conf, cls) in enumerate(zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    class_name = yolo_model.names[int(cls)]
                    
                    detection = {
                        "id": f"det_{i}",
                        "class_name": class_name,
                        "confidence": float(conf),
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "mask": None,  # Simplified for testing
                        "image_size": {"width": frame.shape[1], "height": frame.shape[0]}
                    }
                    
                    # Extract mask if available
                    if r.masks is not None and len(r.masks.data) > i:
                        try:
                            mask = r.masks.data[i].cpu().numpy()
                            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                            contours, _ = cv2.findContours((mask_resized * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                largest_contour = max(contours, key=cv2.contourArea)
                                mask_points = largest_contour.reshape(-1, 2).tolist()
                                detection["mask"] = mask_points
                                logger.info(f"✅ Extracted mask with {len(mask_points)} points")
                        except Exception as e:
                            logger.warning(f"Mask extraction failed: {e}")
                    
                    detections.append(detection)
        
        # Update detection results
        detection_results = {
            "detections": detections,
            "frame_count": 999,  # Special frame number
            "timestamp": time.time()
        }
        
        logger.info(f"🎯 Force detection complete: {len(detections)} objects")
        
        return {
            "status": "success", 
            "detections": detections,
            "count": len(detections)
        }
        
    except Exception as e:
        logger.error(f"Force detection error: {e}")
        return {"status": "error", "message": str(e)}

# =============================================================================
# VIDEO DETECTION ENDPOINTS
# =============================================================================

@app.websocket("/ws-video")
async def video_websocket_endpoint(websocket: WebSocket):
    """
    VIDEO DETECTION WEBSOCKET
    =======================
    Dedicated WebSocket for video detection:
    - Receives video frames one by one
    - Returns detections with both bboxes and masks
    - Maintains video context between frames
    """
    await websocket.accept()
    
    # Parse query parameters from WebSocket URL
    query_params = websocket.query_params
    confidence = float(query_params.get('confidence', 0.3))
    iou_threshold = float(query_params.get('iou_threshold', 0.2))
    
    logger.info(f"🎬 Video WebSocket connected with confidence: {confidence}, IoU: {iou_threshold}")
    
    try:
        frame_count = 0
        start_time = time.time()
        last_fps_log = time.time()
        
        while True:
            # Receive frame data from frontend
            data = await websocket.receive_bytes()
            frame_count += 1
            
            try:
                # Decode the frame
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Convert frame for ONNX (BGR → RGB conversion handled in model)
                    # Note: frame is in BGR format from OpenCV, model will convert it properly
                    
                    # Run FAST live detection for video streaming (non-blocking)
                    # Use the async method directly
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    detections = await onnx_model.fast_live_detect_async(
                        rgb_frame, 
                        conf_threshold=confidence, 
                        iou_threshold=iou_threshold, 
                        max_detections=50
                    )
                    
                            # Log detection count every 30 frames
                    if frame_count % 30 == 0:
                        logger.info(f"🎬 Video frame {frame_count}: {len(detections)} detections")
                    # Convert detections to video format
                    
                    # Convert detections to video format
                    video_detections = []
                    for i, detection in enumerate(detections):
                        video_detection = {
                                    "id": f"det_{frame_count}_{i}",
                            "class_name": detection["class_name"],
                            "confidence": detection["confidence"],
                            "bbox": detection["bbox"],
                            "mask": detection["mask"],
                            "image_size": detection["image_size"]
                        }
                        video_detections.append(video_detection)
                    
                    # Send detection results back
                    response_data = {
                        "detections": video_detections,
                        "frame_number": frame_count,
                        "timestamp": time.time()
                    }
                    await websocket.send_text(json.dumps(response_data))
                    
                    # Log FPS every 5 seconds
                    current_time = time.time()
                    if current_time - last_fps_log >= 5.0:
                        elapsed_time = current_time - start_time
                        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                        logger.info(f"🎬 Video Performance: {fps:.1f} FPS, {frame_count} frames processed")
                        last_fps_log = current_time
            
            except Exception as e:
                logger.error(f"❌ Video frame processing error: {e}")
                error_response = {
                    "detections": [],
                    "frame_number": frame_count,
                    "error": str(e)
                }
                await websocket.send_text(json.dumps(error_response))
    
    except WebSocketDisconnect:
        logger.info("🎬 Video WebSocket disconnected")
    except Exception as e:
        logger.error(f"❌ Video WebSocket error: {e}")

@app.post("/detect-video", response_model=VideoDetectionResponse)
async def detect_video(
    file: UploadFile = File(...),
    confidence: float = 0.5,
    max_frames: int = 100,
    frame_interval: int = 30,  # Process every Nth frame
    motion_threshold: int = 25,  # Threshold for motion detection
    min_motion_area: int = 500,  # Minimum area for motion regions
    keyframe_interval: int = 5  # Run full detection every N frames
):
    """
    VIDEO DETECTION ENDPOINT
    =======================
    Handles video file upload and frame-by-frame detection with optimizations:
    - Motion detection to identify regions of interest
    - Selective YOLO inference on motion regions
    - Full frame detection on keyframes
    - Object tracking between frames
    """
    try:
        start_time = time.time()
        logger.info(f"🎬 Starting optimized video detection for: {file.filename}")
        
        # Check if model is loaded
        if onnx_model is None:
            logger.error("❌ ONNX model not loaded!")
            raise HTTPException(status_code=500, detail="ONNX model not loaded")
        
        # Initialize video processor
        video_processor = VideoProcessor(
            onnx_model=onnx_model,
            confidence_threshold=confidence,
            motion_threshold=motion_threshold,
            min_motion_area=min_motion_area,
            keyframe_interval=keyframe_interval
        )
        
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_video_path = temp_file.name
        
        try:
            # Open video with OpenCV
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Could not open video file")
            
            # Get video info
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            video_info = {
                "total_frames": total_frames,
                "fps": fps,
                "width": width,
                "height": height,
                "duration_seconds": duration
            }
            
            logger.info(f"📹 Video info: {total_frames} frames, {fps:.2f} FPS, {width}x{height}, {duration:.2f}s")
            
            # Calculate frames to process
            frames_to_process = min(max_frames, total_frames)
            logger.info(f"🎯 Will process up to {frames_to_process} frames with optimized detection")
            
            detections_per_frame = []
            processed_frames = 0
            
            while processed_frames < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame with our optimized strategy
                frame_detections = video_processor.process_frame(frame)
                detections_per_frame.append(frame_detections)
                processed_frames += 1
                    
                # Log progress periodically
                if processed_frames % 10 == 0:
                    logger.info(f"📊 Processed {processed_frames}/{frames_to_process} frames - Found {len(frame_detections)} tracked objects")
            
            cap.release()
            processing_time = time.time() - start_time
            
            # Count total detections
            total_detections = sum(len(frame_dets) for frame_dets in detections_per_frame)
            logger.info(f"✅ Video processing complete: {processed_frames} frames, {total_detections} total detections in {processing_time:.2f}s")
            
            return VideoDetectionResponse(
                total_frames=total_frames,
                processed_frames=processed_frames,
                detections_per_frame=detections_per_frame,
                processing_time=processing_time,
                model_version=current_model_name or "Unknown",
                video_info=video_info
            )
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_video_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"❌ Video detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

def detection_worker():
    """Background worker that processes frames and runs YOLO detection"""
    global detection_results
    
    logger.info("🎥 Starting detection worker...")
    
    frame_count = 0
    last_detection_time = 0
    
    while streaming_active:
        try:
            # Process frames from the queue (sent by frontend)
            if not frame_queue.empty():
                try:
                    frame_data = frame_queue.get_nowait()
                    frame_count += 1
                    current_time = time.time()
                    
                    # Decode the frame
                    nparr = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Run ONNX detection on every frame for smooth live detection
                        should_detect = (onnx_model is not None and 
                                       current_time - last_detection_time > 0.05)  # Max 20 FPS detection
                        
                        if should_detect:
                            try:
                                last_detection_time = current_time
                                
                                # Convert frame for ONNX (optimized)
                                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                
                                # Run ULTRA-FAST live detection for background worker
                                detections = onnx_model.fast_live_detect(rgb_frame, conf_threshold=0.25, max_detections=50)
                                
                                # Update detection results for WebSocket
                                detection_results = {
                                    "detections": detections,
                                    "frame_count": frame_count,
                                    "timestamp": current_time
                                }
                                
                                if detections:
                                    logger.info(f"🎯 Live: {len(detections)} objects detected")
                                
                            except Exception as e:
                                logger.error(f"Detection error: {e}")
                    else:
                        logger.error(f"❌ Failed to decode frame {frame_count}")
                
                except queue.Empty:
                    pass
                except Exception as e:
                    logger.error(f"Frame processing error: {e}")
            else:
                # Small delay to prevent busy waiting when no frames
                time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Worker loop error: {e}")
            time.sleep(0.1)
    
    logger.info("🎥 Detection worker stopped") 