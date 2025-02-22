import asyncio
import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

from backend.preprocessing.noise_masking import NoiseReducer


class VideoReconstructionAgent:
    def __init__(self):
        self.ideal_metrics = {
            'elbow_angle': 85,  # Degrees - FIBA recommended shooting form
            'wrist_angle': 75,  # Degrees - for optimal follow through
            'shoulder_angle': 90,  # Degrees - squared to basket
            'knee_angle': 115,  # Degrees - for proper leg drive
            'shot_trajectory': 52,  # Degrees - optimal arc (research-backed)
            'release_height_ratio': 1.3,  # Above head release point
        }
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def generate_ideal_form_video(self, input_video_path, output_path):
        """Generate visualization of ideal form using user's video as reference"""
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
        
        frames = []
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get pose landmarks
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Create visualization frame
                viz_frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Draw original pose
                self.mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
                
                # Generate and draw ideal pose
                ideal_landmarks = self._generate_ideal_pose(results.pose_landmarks)
                self._draw_ideal_pose(viz_frame, ideal_landmarks)
                
                # Add annotations
                self._add_annotations(viz_frame, ideal_landmarks)
                
                # Combine original and visualization frames
                combined_frame = np.hstack([frame, viz_frame])
                out.write(combined_frame)
                
        cap.release()
        out.release()
        return output_path

    def _generate_ideal_pose(self, original_landmarks):
        """Generate ideal pose landmarks based on original pose"""
        # Create a deep copy of the original landmarks
        ideal_landmarks = type(original_landmarks)()
        ideal_landmarks.CopyFrom(original_landmarks)
        
        # Adjust key angles based on ideal metrics
        # Elbow adjustment
        self._adjust_joint_angle(
            ideal_landmarks,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.ideal_metrics['elbow_angle']
        )
        
        # Knee adjustment
        self._adjust_joint_angle(
            ideal_landmarks,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.ideal_metrics['knee_angle']
        )
        
        return ideal_landmarks

    def _adjust_joint_angle(self, landmarks, joint1, joint2, joint3, target_angle):
        """Adjust joint angle to match target angle"""
        # Get current angle
        current_angle = self._calculate_angle(
            landmarks.landmark[joint1],
            landmarks.landmark[joint2],
            landmarks.landmark[joint3]
        )
        
        # Calculate rotation needed
        rotation = target_angle - current_angle
        
        # Apply rotation to joint3 around joint2
        self._rotate_point(
            landmarks.landmark[joint3],
            landmarks.landmark[joint2],
            rotation
        )

    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        
        angle = np.degrees(
            np.arccos(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            )
        )
        return angle

    def _rotate_point(self, point, center, angle):
        """Rotate a point around a center point by given angle"""
        angle_rad = np.radians(angle)
        
        # Translate point to origin
        translated = np.array([point.x - center.x, point.y - center.y])
        
        # Rotate
        rotated = np.array([
            translated[0] * np.cos(angle_rad) - translated[1] * np.sin(angle_rad),
            translated[0] * np.sin(angle_rad) + translated[1] * np.cos(angle_rad)
        ])
        
        # Translate back
        point.x = rotated[0] + center.x
        point.y = rotated[1] + center.y

    def _draw_ideal_pose(self, frame, landmarks):
        """Draw ideal pose with custom styling"""
        self.mp_draw.draw_landmarks(
            frame,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_draw.DrawingSpec(
                color=(0, 255, 0),
                thickness=2,
                circle_radius=2
            ),
            connection_drawing_spec=self.mp_draw.DrawingSpec(
                color=(255, 255, 255),
                thickness=1
            )
        )

    def _add_annotations(self, frame, landmarks):
        """Add angle measurements and guidelines to visualization"""
        # Add angle measurements
        angles = {
            'Elbow': self._calculate_angle(
                landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
                landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW],
                landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            ),
            'Knee': self._calculate_angle(
                landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP],
                landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE],
                landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            )
        }
        
        y_offset = 30
        for i, (joint, angle) in enumerate(angles.items()):
            cv2.putText(
                frame,
                f"{joint} Angle: {angle:.1f}°",
                (10, y_offset + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

    def get_form_feedback(self, shot_metrics):
        """Generate detailed feedback on form adjustments needed"""
        # Check if shot_metrics is empty or None
        if not shot_metrics:
            return {
                'adjustments': {},
                'summary': ["Unable to analyze shot form. Please ensure the video shows a complete basketball shot."]
            }
        
        # Process metrics directly since it's a single dictionary
        adjustments = {}
        for metric, ideal_value in self.ideal_metrics.items():
            if metric in shot_metrics:  # Remove the [0] index access
                try:
                    current = float(shot_metrics[metric])
                    if current > 0:  # Only process non-zero metrics
                        diff = ideal_value - current
                        adjustments[metric] = {
                            'current': current,
                            'ideal': ideal_value,
                            'adjustment': diff
                        }
                except (ValueError, TypeError):
                    print(f"Warning: Invalid value for metric {metric}: {shot_metrics[metric]}")
                    continue
        
        feedback = {
            'adjustments': adjustments,
            'summary': []
        }
        
        if adjustments:
            for metric, adj in adjustments.items():
                if abs(adj['adjustment']) > 5:  # Significant adjustment needed
                    feedback['summary'].append(
                        f"Adjust {metric.replace('_', ' ')} by {abs(adj['adjustment']):.1f} degrees"
                    )
        else:
            feedback['summary'].append("Form analysis incomplete. Try recording from a different angle for better analysis.")
        
        return feedback

    def process_video(self, input_video_path, output_path=None):
        """Main entry point for video processing and analysis"""
        if output_path is None:
            output_path = str(Path(input_video_path).parent / "ideal_form.mp4")
            
        # Check if input video exists
        if not Path(input_video_path).exists():
            raise FileNotFoundError(f"Input video not found: {input_video_path}")
            
        # Generate ideal form video
        processed_video_path = self.generate_ideal_form_video(input_video_path, output_path)
        
        # Get metrics from the video
        metrics = self._extract_metrics_from_video(input_video_path)
        
        # Generate feedback
        feedback = self.get_form_feedback(metrics)
        
        return {
            'video_path': processed_video_path,
            'metrics': metrics,
            'feedback': feedback
        }
        
   
    def _extract_metrics_from_video(self, video_path):
        """Extract metrics from key frames of the video"""
        metrics = {
            'elbow_angle': 0,
            'wrist_angle': 0,
            'shoulder_angle': 0,
            'knee_angle': 0,
            'shot_trajectory': 0,
            'release_height_ratio': 0
        }
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Process every 5th frame
            if frame_count % 5 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)
                
                if results.pose_landmarks:
                    # Calculate angles
                    elbow_angle = self._calculate_angle(
                        results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
                        results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW],
                        results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
                    )
                    
                    knee_angle = self._calculate_angle(
                        results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP],
                        results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE],
                        results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
                    )
                    
                    # Update metrics with max angles (assuming shooting motion)
                    metrics['elbow_angle'] = max(metrics['elbow_angle'], elbow_angle)
                    metrics['knee_angle'] = max(metrics['knee_angle'], knee_angle)
                    
            frame_count += 1
        
        cap.release()
        return metrics


def get_reconstructed_video(input_video: str, output_path: str):
    """
    Process video and return metrics
    Args:
        input_video: Path to input video file
        output_path: Path to save output video
    Returns:
        Dictionary containing metrics and paths to both masked and reconstructed videos
    """
    # Get absolute paths
    input_path = str(Path(input_video).resolve())
    
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate paths for both masked and reconstructed videos
    masked_path = str(output_dir / f"masked_{Path(output_path).name}")
    reconstructed_path = str(output_dir / f"{Path(output_path).name}")

    print(f"Processing video from: {input_path}")
    print(f"Masked video will be saved to: {masked_path}")
    print(f"Reconstructed video will be saved to: {reconstructed_path}")

    # First process video with noise reduction
    noise_reducer = NoiseReducer()
    asyncio.run(noise_reducer.process_video(input_path, masked_path))

    # Then generate reconstructed video with pose analysis
    agent = VideoReconstructionAgent()
    result = agent.process_video(masked_path, reconstructed_path)
    
    print("\nProcessing Complete!")
    print(f"Masked video saved to: {masked_path}")
    print(f"Reconstructed video saved to: {result['video_path']}")
    
    print("\nMetrics:")
    for metric, value in result['metrics'].items():
        print(f"{metric}: {value:.1f}°")
    
    print("\nFeedback Summary:")
    for feedback in result['feedback']['summary']:
        print(f"- {feedback}")

    return {
        'metrics': result['metrics'],
        'masked_video_path': masked_path,
        'reconstructed_video_path': result['video_path'],
        'feedback': result['feedback']
    }

if __name__ == "__main__":
    # Example usage with custom output path
    result = get_reconstructed_video(
        input_video="noisy_images/clip_001.mp4",
        output_path="noisy_images/output.mp4"
    )
