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
    def __init__(self, onnx_model, confidence_threshold: float = 0.5,
                 motion_threshold: int = 25,
                 min_motion_area: int = 500,
                 keyframe_interval: int = 5):
        """
        Initialize the video processor with detection and tracking parameters
        
        Args:
            onnx_model: ONNX model instance
            confidence_threshold: Confidence threshold for ONNX detections
            motion_threshold: Threshold for motion detection (0-255)
            min_motion_area: Minimum area for motion regions
            keyframe_interval: Run full detection every N frames
        """
        self.onnx_model = onnx_model
        self.confidence_threshold = confidence_threshold
        self.motion_threshold = motion_threshold
        self.min_motion_area = min_motion_area
        self.keyframe_interval = keyframe_interval
        
        # State variables
        self.prev_frame = None
        self.frame_count = 0
        self.tracked_objects: Dict[str, TrackedObject] = {}
        self.next_track_id = 0
    
    def _detect_motion_regions(self, current_frame: np.ndarray) -> List[MotionRegion]:
        """
        Detect regions of motion between current and previous frame
        
        Args:
            current_frame: Current frame as numpy array
            
        Returns:
            List of MotionRegion objects
        """
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return []
            
        # Convert current frame to grayscale
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Compute absolute difference and threshold
        frame_diff = cv2.absdiff(self.prev_frame, current_gray)
        _, motion_mask = cv2.threshold(frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to remove noise and connect regions
        kernel = np.ones((5,5), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of motion regions
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and convert contours to MotionRegion objects
        motion_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_motion_area:
                x, y, w, h = cv2.boundingRect(contour)
                motion_regions.append(MotionRegion(
                    x1=x, y1=y,
                    x2=x+w, y2=y+h,
                    area=area
                ))
        
        # Update previous frame
        self.prev_frame = current_gray
        return motion_regions
    
    def _region_overlaps_tracked_object(self, region: MotionRegion) -> bool:
        """Check if a motion region overlaps with any tracked object"""
        for obj in self.tracked_objects.values():
            bbox = obj.bbox
            # Check for overlap
            if not (region.x2 < bbox['x1'] or
                   region.x1 > bbox['x2'] or
                   region.y2 < bbox['y1'] or
                   region.y1 > bbox['y2']):
                return True
        return False
    
    def _crop_region(self, frame: np.ndarray, region: MotionRegion,
                    padding: int = 20) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Crop a region from frame with padding, handling boundaries
        Returns the crop and the offset (for translating coordinates back)
        """
        h, w = frame.shape[:2]
        x1 = max(0, region.x1 - padding)
        y1 = max(0, region.y1 - padding)
        x2 = min(w, region.x2 + padding)
        y2 = min(h, region.y2 + padding)
        
        return frame[y1:y2, x1:x2], (x1, y1)
    
    def _update_tracked_objects(self, frame: np.ndarray,
                              detections: List[Dict],
                              frame_number: int):
        """Update tracked objects with new detections"""
        # Mark all existing tracks as unmatched
        unmatched_tracks = set(self.tracked_objects.keys())
        
        # Update existing tracks with new detections
        for det in detections:
            best_iou = 0
            best_track_id = None
            
            # Find best matching track by IoU
            for track_id in unmatched_tracks:
                track = self.tracked_objects[track_id]
                iou = self._calculate_iou(det['bbox'], track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_iou > 0.3:  # IoU threshold for matching
                # Update existing track
                track = self.tracked_objects[best_track_id]
                track.bbox = det['bbox']
                track.confidence = det['confidence']
                track.mask = det.get('mask')
                track.track_count += 1
                track.last_full_detect = frame_number
                unmatched_tracks.remove(best_track_id)
            else:
                # Create new track
                new_id = f"track_{self.next_track_id}"
                self.next_track_id += 1
                self.tracked_objects[new_id] = TrackedObject(
                    id=new_id,
                    class_name=det['class_name'],
                    confidence=det['confidence'],
                    bbox=det['bbox'],
                    mask=det.get('mask'),
                    track_count=1,
                    last_full_detect=frame_number
                )
        
        # Remove unmatched tracks that haven't been detected recently
        for track_id in unmatched_tracks:
            if frame_number - self.tracked_objects[track_id].last_full_detect > 10:
                del self.tracked_objects[track_id]
    
    def _calculate_iou(self, bbox1: Dict[str, int], bbox2: Dict[str, int]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1 = max(bbox1['x1'], bbox2['x1'])
        y1 = max(bbox1['y1'], bbox2['y1'])
        x2 = min(bbox1['x2'], bbox2['x2'])
        y2 = min(bbox1['y2'], bbox2['y2'])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        
        area1 = (bbox1['x2'] - bbox1['x1']) * (bbox1['y2'] - bbox1['y1'])
        area2 = (bbox2['x2'] - bbox2['x1']) * (bbox2['y2'] - bbox2['y1'])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Process a single frame with our optimized detection strategy
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of detection dictionaries with bbox, confidence, etc.
        """
        self.frame_count += 1
        frame_detections = []
        
        # Step 1: Detect motion regions
        motion_regions = self._detect_motion_regions(frame)
        
        # Step 2: Process motion regions that don't overlap with tracked objects
        for region in motion_regions:
            if not self._region_overlaps_tracked_object(region):
                # Crop region and run detection
                crop, (offset_x, offset_y) = self._crop_region(frame, region)
                if crop.size > 0:  # Ensure valid crop
                    # Convert crop to PIL Image for ONNX model
                    from PIL import Image
                    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    detections = self.onnx_model.predict(crop_pil, conf_threshold=self.confidence_threshold)
                    
                    for detection in detections:
                        # Translate coordinates back to full frame
                        bbox = detection['bbox']
                        translated_detection = {
                            'class_name': detection['class_name'],
                            'confidence': detection['confidence'],
                            'bbox': {
                                'x1': bbox['x1'] + offset_x,
                                'y1': bbox['y1'] + offset_y,
                                'x2': bbox['x2'] + offset_x,
                                'y2': bbox['y2'] + offset_y
                            },
                            'mask': detection.get('mask')
                        }
                        frame_detections.append(translated_detection)
        
        # Step 3: Run full frame detection on keyframes
        if self.frame_count % self.keyframe_interval == 0:
            # Convert frame to PIL Image for ONNX model
            from PIL import Image
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detections = self.onnx_model.predict(frame_pil, conf_threshold=self.confidence_threshold)
            
            for detection in detections:
                frame_detections.append(detection)
        
        # Step 4: Update object tracking
        self._update_tracked_objects(frame, frame_detections, self.frame_count)
        
        # Return current tracked objects as detections
        return [
            {
                'id': obj.id,
                'class_name': obj.class_name,
                'confidence': obj.confidence,
                'bbox': obj.bbox,
                'mask': obj.mask,
                'track_count': obj.track_count
            }
            for obj in self.tracked_objects.values()
        ] 