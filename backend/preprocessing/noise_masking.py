import os

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO


class NoiseReducer:
    def __init__(self):
        # Initialize YOLO model for ball and hoop detection
        self.model = YOLO("backend/runs/detect/train/weights/best.pt")
        
        # Initialize MediaPipe for pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.last_shooter_mask = None
        self.shooter_detected = False
        self.persistence_frames = 30  # Number of frames to persist the shooter mask
        self.persistence_counter = 0

    def detect_shooter(self, pose_results, ball_pos):
        """Detect if pose is the shooter based on proximity to ball"""
        if not pose_results.pose_landmarks or not ball_pos:
            return False
        
        # Get right wrist position (landmark 16)
        h, w, _ = self.frame_shape
        wrist = pose_results.pose_landmarks.landmark[16]
        wrist_pos = (int(wrist.x * w), int(wrist.y * h))
        
        # Calculate distance to ball
        ball_center = (ball_pos[0] + ball_pos[2] // 2, ball_pos[1] + ball_pos[3] // 2)
        distance = np.sqrt((wrist_pos[0] - ball_center[0])**2 + (wrist_pos[1] - ball_center[1])**2)
        
        # More generous distance threshold
        return distance < 100  # Increased from 50 to 100

    def create_mask(self, frame, results, pose_results):
        self.frame_shape = frame.shape
        mask = np.zeros_like(frame)
        ball_detected = False
        ball_pos = None
        
        # Process YOLO detections (ball and hoop)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get coordinates and class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                
                # If this is a basketball, store its position
                if cls == 0:  # Basketball class
                    ball_pos = (x1, y1, x2-x1, y2-y1)
                    ball_detected = True
                    
                    # Draw ball on mask with larger padding
                    padding = 50  # Increased from 30 to 50
                    cv2.rectangle(mask, 
                                (max(0, x1-padding), max(0, y1-padding)),
                                (min(frame.shape[1], x2+padding), min(frame.shape[0], y2+padding)),
                                (255, 255, 255), -1)
                elif cls == 1:  # Hoop class
                    # Always draw the hoop with larger padding
                    padding = 50  # Increased from 30 to 50
                    cv2.rectangle(mask, 
                                (max(0, x1-padding), max(0, y1-padding)),
                                (min(frame.shape[1], x2+padding), min(frame.shape[0], y2+padding)),
                                (255, 255, 255), -1)

        # Process pose landmarks only if we have detected a ball
        if pose_results.pose_landmarks and ball_pos:
            h, w, _ = frame.shape
            
            # Check if this pose is the shooter
            is_shooter = self.detect_shooter(pose_results, ball_pos)
            
            if is_shooter or self.shooter_detected:
                points = []
                # Create points array for shooter's pose with larger offset
                for landmark in pose_results.pose_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    
                    # Increased offset for wider coverage
                    offset = 40  # Increased from 20 to 40
                    points.extend([
                        [x - offset, y - offset],
                        [x + offset, y - offset],
                        [x - offset, y + offset],
                        [x + offset, y + offset],
                        [x, y],
                        # Add additional points for better coverage
                        [x - offset*1.5, y],
                        [x + offset*1.5, y],
                        [x, y - offset*1.5],
                        [x, y + offset*1.5]
                    ])
                
                if points:
                    points = np.array(points, np.int32)
                    hull = cv2.convexHull(points)
                    # Draw the hull with additional padding
                    cv2.fillPoly(mask, [hull], (255, 255, 255))
                    
                    # Store this mask for the shooter
                    self.last_shooter_mask = mask.copy()
                    self.shooter_detected = True
                    self.persistence_counter = self.persistence_frames
        
        # If we have a stored shooter mask and no current shooter
        if not ball_detected and self.last_shooter_mask is not None and self.persistence_counter > 0:
            # Use the stored shooter mask
            mask = cv2.bitwise_or(mask, self.last_shooter_mask)
            self.persistence_counter -= 1
            
            # If persistence has expired, reset
            if self.persistence_counter == 0:
                self.last_shooter_mask = None
                self.shooter_detected = False

        # Final dilation with larger kernel
        kernel_size = 35  # Increased from 25 to 35
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)  # Increased iterations from 2 to 3
        
        # Optional: Add a slight blur to smooth the mask edges
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        return mask

    async def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            print(f"Processing frame {frame_count}")
            
            # Run YOLO detection (without stream=True)
            results = self.model(frame)
            
            # Run pose detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(frame_rgb)
            
            # Create mask
            mask = self.create_mask(frame, results, pose_results)
            
            # Apply mask to original frame
            masked_frame = cv2.bitwise_and(frame, mask)
            
            # Optional: Add slight blur to smooth edges
            masked_frame = cv2.GaussianBlur(masked_frame, (3, 3), 0)
            
            # Write frame
            out.write(masked_frame)
            
            # Display frame (optional, for debugging)
            cv2.imshow('Masked Frame', masked_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()

def main():
    noise_reducer = NoiseReducer()
    
    # Get project root directory by going up one level from preprocessing folder
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Construct full paths using project root
    input_video = os.path.join(project_root, "noisy_images", "clip_023.mp4")
    output_video = os.path.join(project_root, "noisy_images", "masked_shot.mp4")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    
    # Debug print
    print(f"Looking for input video at: {input_video}")
    
    # Check if input file exists
    if not os.path.exists(input_video):
        print(f"Error: Input video file not found at: {input_video}")
        return
        
    print(f"Processing video: {input_video}")
    print(f"Output will be saved to: {output_video}")
    
    noise_reducer.process_video(input_video, output_video)

if __name__ == "__main__":
    main()