import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class MotionRegion:
    """Represents a region of detected motion in a frame"""
    x1: int
    y1: int
    x2: int
    y2: int
    area: float

@dataclass
class TrackedObject:
    """Represents a tracked object across frames"""
    id: str
    class_name: str
    confidence: float
    bbox: Dict[str, int]
    mask: Optional[List[List[float]]] = None
    track_count: int = 0
    last_full_detect: int = 0

class VideoProcessor:
    """Video processing class that handles both ONNX and PyTorch models"""
    
    def __init__(self, model, confidence_threshold=0.6, motion_threshold=0.3, max_detections=50):
        """Initialize video processor
        
        Args:
            model: Either SimpleONNXModel or UnifiedModel instance
            confidence_threshold: Detection confidence threshold
            motion_threshold: Motion detection threshold
            max_detections: Maximum number of detections per frame
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.motion_threshold = motion_threshold
        self.max_detections = max_detections
        
        # Initialize motion detection
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        
        # Initialize frame tracking
        self.frame_count = 0
        self.total_processing_time = 0
        self.last_frame_time = None
        
        logger.info(f"✅ Video processor initialized with {type(model).__name__}")
        logger.info(f"   - Confidence threshold: {confidence_threshold}")
        logger.info(f"   - Motion threshold: {motion_threshold}")
        logger.info(f"   - Max detections: {max_detections}")
    
    async def process_frame(self, frame: np.ndarray) -> Tuple[List[dict], bool]:
        """Process a single frame
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            Tuple of (detections, has_motion)
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
            # Check for motion
            has_motion = self._detect_motion(frame)
        
            # Run detection if there's motion
            if has_motion:
                # Use fast_mode=True for streaming
                detections = self.model.predict(
                    rgb_frame,
                    conf_threshold=self.confidence_threshold,
                    iou_threshold=self.motion_threshold,
                    fast_mode=True
                )
                
                # Update statistics
                self.frame_count += 1
                current_time = time.time()
                if self.last_frame_time:
                    self.total_processing_time += current_time - self.last_frame_time
                self.last_frame_time = current_time
                
                return detections, True
            
            return [], False
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return [], False
    
    def _detect_motion(self, frame: np.ndarray) -> bool:
        """Detect motion in frame using background subtraction
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            True if motion detected, False otherwise
        """
        try:
            # Apply background subtraction
            fgmask = self.fgbg.apply(frame)
        
            # Calculate motion percentage
            motion_pixels = np.sum(fgmask > 0)
            total_pixels = fgmask.size
            motion_percentage = motion_pixels / total_pixels
            
            return motion_percentage > self.motion_threshold
            
        except Exception as e:
            logger.error(f"Motion detection error: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get processing statistics
        
        Returns:
            Dictionary with processing statistics
        """
        if self.frame_count > 0:
            avg_time = self.total_processing_time / self.frame_count
            fps = 1 / avg_time if avg_time > 0 else 0
        else:
            fps = 0
            
        return {
            "frames_processed": self.frame_count,
            "total_time": self.total_processing_time,
            "average_fps": fps
        } 