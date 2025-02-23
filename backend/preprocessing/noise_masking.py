import asyncio
import logging
import os
from dataclasses import dataclass
import subprocess

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO


@dataclass
class MaskingConfig:
    ball_padding: int = 50
    hoop_padding: int = 50
    pose_offset: int = 40
    kernel_size: int = 35
    shooter_distance_threshold: int = 100
    persistence_frames: int = 30
    frame_smooth_alpha: float = 0.7
    process_every_n_frames: int = 1
    dilation_iterations: int = 3
    frame_delay: float = 0  # Delay between frames in seconds

class NoiseReducer:
    def __init__(self, config: MaskingConfig = None):
        self.config = config or MaskingConfig()
        
        # Initialize YOLO model for ball and hoop detection
        self.model = YOLO("runs/detect/train/weights/best.pt")
        
        # Initialize MediaPipe for pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.last_shooter_mask = None
        self.last_valid_mask = None
        self.shooter_detected = False
        self.persistence_counter = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def detect_shooter(self, pose_results, ball_pos):
        """Enhanced shooter detection using multiple joint positions"""
        if not pose_results.pose_landmarks or not ball_pos:
            return False
        
        h, w, _ = self.frame_shape
        
        # Check both wrists and elbows
        relevant_landmarks = [
            15,  # Left wrist
            16,  # Right wrist
            13,  # Left elbow
            14   # Right elbow
        ]
        
        ball_center = (ball_pos[0] + ball_pos[2] // 2, ball_pos[1] + ball_pos[3] // 2)
        
        min_distance = float('inf')
        for landmark_idx in relevant_landmarks:
            landmark = pose_results.pose_landmarks.landmark[landmark_idx]
            pos = (int(landmark.x * w), int(landmark.y * h))
            distance = np.sqrt((pos[0] - ball_center[0])**2 + (pos[1] - ball_center[1])**2)
            min_distance = min(min_distance, distance)
        
        return min_distance < self.config.shooter_distance_threshold

    def smooth_mask(self, current_mask):
        """Smooth transition between consecutive masks"""
        if self.last_valid_mask is None:
            self.last_valid_mask = current_mask
            return current_mask
        
        smoothed_mask = cv2.addWeighted(
            current_mask,
            self.config.frame_smooth_alpha,
            self.last_valid_mask,
            1 - self.config.frame_smooth_alpha,
            0
        )
        self.last_valid_mask = smoothed_mask
        return smoothed_mask

    def create_mask(self, frame, results, pose_results):
        """Create mask with improved error handling and smoothing"""
        try:
            self.frame_shape = frame.shape
            mask = np.zeros_like(frame)
            ball_detected = False
            ball_pos = None
            
            # Process YOLO detections
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    
                    padding = self.config.ball_padding if cls == 0 else self.config.hoop_padding
                    
                    if cls == 0:  # Basketball
                        ball_pos = (x1, y1, x2-x1, y2-y1)
                        ball_detected = True
                    
                    # Draw padded rectangle for both ball and hoop
                    cv2.rectangle(mask, 
                                (max(0, x1-padding), max(0, y1-padding)),
                                (min(frame.shape[1], x2+padding), min(frame.shape[0], y2+padding)),
                                (255, 255, 255), -1)

            # Process pose landmarks
            if pose_results.pose_landmarks and ball_pos:
                is_shooter = self.detect_shooter(pose_results, ball_pos)
                
                if is_shooter or self.shooter_detected:
                    self._add_pose_to_mask(mask, pose_results)
                    self.last_shooter_mask = mask.copy()
                    self.shooter_detected = True
                    self.persistence_counter = self.config.persistence_frames

            # Handle persistence
            if not ball_detected and self.last_shooter_mask is not None and self.persistence_counter > 0:
                mask = cv2.bitwise_or(mask, self.last_shooter_mask)
                self.persistence_counter -= 1
                
                if self.persistence_counter == 0:
                    self.last_shooter_mask = None
                    self.shooter_detected = False

            # Final processing
            kernel = np.ones((self.config.kernel_size, self.config.kernel_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=self.config.dilation_iterations)
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            
            # Apply temporal smoothing
            mask = self.smooth_mask(mask)
            
            return mask

        except Exception as e:
            self.logger.error(f"Error creating mask: {str(e)}")
            return np.ones_like(frame) * 255  # Return full white mask as fallback

    def _add_pose_to_mask(self, mask, pose_results):
        """Helper method to add pose landmarks to mask"""
        h, w, _ = self.frame_shape
        points = []
        
        for landmark in pose_results.pose_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            offset = self.config.pose_offset
            
            points.extend([
                [x - offset, y - offset],
                [x + offset, y - offset],
                [x - offset, y + offset],
                [x + offset, y + offset],
                [x, y],
                [x - offset*1.5, y],
                [x + offset*1.5, y],
                [x, y - offset*1.5],
                [x, y + offset*1.5]
            ])
        
        if points:
            points = np.array(points, np.int32)
            hull = cv2.convexHull(points)
            cv2.fillPoly(mask, [hull], (255, 255, 255))

    async def process_video(self, input_path, output_path):
        """Process video and save masked version"""
        try:
            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open input video: {input_path}")

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Create video writer with avc1 codec
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Changed from mp4v to avc1
            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height),
                isColor=True
            )

            if not writer.isOpened():
                # Fallback to mp4v codec if avc1 fails
                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (width, height),
                    isColor=True
                )

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                results = self.model(frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = self.pose.process(frame_rgb)
                
                # Create and apply mask
                mask = self.create_mask(frame, results, pose_results)
                masked_frame = cv2.bitwise_and(frame, mask)
                
                # Write frame
                writer.write(masked_frame)

            # Clean up
            cap.release()
            writer.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            if 'cap' in locals():
                cap.release()
            if 'writer' in locals():
                writer.release()
            cv2.destroyAllWindows()
            raise

