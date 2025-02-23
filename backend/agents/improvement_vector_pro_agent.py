import os
import sys
import time
from pathlib import Path
from typing import List, Dict
# Add both backend and project root to Python path
backend_dir = str(Path(__file__).parent.parent.parent)
project_root = str(Path(__file__).parent.parent.parent.parent)

# Add paths if they're not already there
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set PYTHONPATH environment variable
os.environ['PYTHONPATH'] = f"{project_root}:{backend_dir}"

# Now import the modules
from shot_detector import ShotDetector
from preprocessing.noise_masking import NoiseReducer

import numpy as np
import asyncio

class ImprovementVectorProAgent:
    def __init__(self):
        self.feature_weights = {
            'elbow_angle': 0.2,
            'wrist_angle': 0.15,
            'shoulder_angle': 0.2,
            'knee_angle': 0.15,
            'shot_trajectory': 0.2,
            'release_height_ratio': 0.1
        }

    def normalize_metrics(self, metrics):
        """Normalize metrics to sum to desired total"""
        # Filter out non-metric keys like 'video_url'
        metric_values = {k: float(v) for k, v in metrics.items() if k in self.feature_weights}
        
        # Don't normalize if metrics are already properly scaled
        total = sum(metric_values.values())
        if abs(total - 50) < 0.1 or abs(total - 100) < 0.1:
            return metric_values
        
        # Otherwise normalize to 100
        if total > 0:
            scale_factor = 100 / total
            return {k: v * scale_factor for k, v in metric_values.items()}
        return {k: 0 for k in self.feature_weights}

    def calculate_improvement_vector(self, video_sequence):
        """
        Calculate improvement vector from a sequence of 3 videos
        video_sequence: List of dicts containing video paths and metrics
        Returns: Improvement vector and score for middle video
        """
        if len(video_sequence) != 3:
            raise ValueError("Expected 3 videos in sequence")

        try:
            # Extract metrics, excluding video_url
            start_metrics = {k: v for k, v in video_sequence[0].items() if k in self.feature_weights}
            current_metrics = {k: v for k, v in video_sequence[1].items() if k in self.feature_weights}
            target_metrics = {k: v for k, v in video_sequence[2].items() if k in self.feature_weights}

            # Normalize metrics
            start_metrics_norm = self.normalize_metrics(start_metrics)
            current_metrics_norm = self.normalize_metrics(current_metrics)
            target_metrics_norm = self.normalize_metrics(target_metrics)

            # Calculate improvement vector
            improvement_vector = {}
            for metric in self.feature_weights:
                start_val = start_metrics_norm.get(metric, 0)
                target_val = target_metrics_norm.get(metric, 0)
                improvement_vector[metric] = target_val - start_val

            # Calculate score
            score = self.calculate_progress_score(
                start_metrics_norm,
                current_metrics_norm,
                target_metrics_norm
            )

            return {
                'improvement_vector': improvement_vector,
                'current_score': score,
                'status': 'success'
            }

        except Exception as e:
            print(f"Error in calculate_improvement_vector: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def calculate_progress_score(self, start, current, target):
        """Calculate progress score (0-100) based on position along improvement vector"""
        total_score = 0
        total_weight = 0

        for metric, weight in self.feature_weights.items():
            if metric not in start or metric not in current or metric not in target:
                continue

            start_val = float(start[metric])
            current_val = float(current[metric])
            target_val = float(target[metric])

            # Calculate progress relative to the total improvement needed
            total_improvement_needed = target_val - start_val
            if abs(total_improvement_needed) > 0.001:  # Avoid division by very small numbers
                current_improvement = current_val - start_val
                progress = current_improvement / total_improvement_needed
                # Clip progress to 0-1 range
                progress = max(0, min(1, progress))
                total_score += progress * weight
                total_weight += weight

        if total_weight == 0:
            return 0

        return (total_score / total_weight) * 100

    def analyze_sequence(self, video_sequence):
        """
        Main entry point for analysis
        video_sequence: List of dicts with 'video_url' and metrics
        """
        try:
            result = self.calculate_improvement_vector(video_sequence)
            
            # Format improvement vector for readability
            formatted_vector = {
                metric: f"{value:.2f}" 
                for metric, value in result['improvement_vector'].items()
            }

            return {
                'improvement_vector': formatted_vector,
                'current_score': round(result['current_score'], 2),
                'status': 'success'
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }


def fetch_metadata_for_video(input_video, output_video):
    # Create unique output path for each video using timestamp
    timestamp = int(time.time() * 1000)
    unique_output = output_video.replace('.mp4', f'_{timestamp}.mp4')
    
    # Debug prints
    print(f"Looking for input video at: {input_video}")
    print(f"Output will be saved to: {unique_output}")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(unique_output), exist_ok=True)
        
        # Check if input file exists
        if not os.path.exists(input_video):
            print(f"Error: Input video file not found at: {input_video}")
            raise FileNotFoundError(f"Input video not found: {input_video}")
        
        noise_reducer = NoiseReducer()
        
        # Run noise reduction asynchronously and wait for it to complete
        asyncio.run(noise_reducer.process_video(input_video, unique_output))
        
        print("MASKED SHOT CREATED")
        
        # Now that noise reduction is complete, run shot detection
        detector = ShotDetector()  
        shot_metrics = detector.shot_metrics
        print("Tried with mask ", shot_metrics)
        
        # Call without the mask if needed
        if not shot_metrics:  
            detector = ShotDetector(unmasked_video_path=input_video)
            shot_metrics = detector.shot_metrics
            print('SHOT METRICS FROM DETECTOR without mask', detector.shot_metrics)

        # Cleanup temporary file
        if os.path.exists(unique_output):
            os.remove(unique_output)

        # Return the first metrics dictionary from the list
        if shot_metrics and len(shot_metrics) > 0:
            return shot_metrics[0]  # Return first element since it's a list
        
    except Exception as e:
        print(f"Error in fetch_metadata_for_video: {str(e)}")
        raise e
    finally:
        # Ensure cleanup happens even if there's an error
        if os.path.exists(unique_output):
            os.remove(unique_output)
    
    # Return empty metrics if nothing was detected
    return {
        'elbow_angle': 0.0,
        'wrist_angle': 0.0,
        'shoulder_angle': 0.0,
        'knee_angle': 0.0,
        'shot_trajectory': 0.0,
        'release_height_ratio': 0.0
    }

def generate_metadata_from_videos(list_videos_urls):
    """Given a list of video urls generate a list of metadata for each object"""
    metadata_list = []
    
    for video_url in list_videos_urls:
        try:
            print(f"\nProcessing video: {video_url}")
            metrics = fetch_metadata_for_video(
                video_url, 
                'backend/noisy_images/temp_masked_shot.mp4'
            )
            
            print("GENERATED METRIC FOR VIDEO ", video_url, ":", metrics)
            
            # Create metadata object with relevant metrics
            metadata = {
                'video_url': video_url,
                'elbow_angle': float(metrics['elbow_angle']),
                'wrist_angle': float(metrics['wrist_angle']),
                'shoulder_angle': float(metrics['shoulder_angle']),
                'knee_angle': float(metrics['knee_angle']),
                'shot_trajectory': float(metrics['shot_trajectory']),
                'release_height_ratio': float(metrics['release_height_ratio'])
            }
            
            metadata_list.append(metadata)
            print(f"Processed {video_url} successfully")
            
            # Add small delay between videos
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing video {video_url}: {str(e)}")
            # Add empty metrics for failed video
            metadata_list.append({
                'video_url': video_url,
                'elbow_angle': 0.0,
                'wrist_angle': 0.0,
                'shoulder_angle': 0.0,
                'knee_angle': 0.0,
                'shot_trajectory': 0.0,
                'release_height_ratio': 0.0
            })
            
    return metadata_list


def get_pro_improvement_vector(test_videos: List) -> Dict:
    # Generate metadata
    metadata_list = generate_metadata_from_videos(test_videos)
    
    # Print results
    print("\nGenerated Metadata:")
    for metadata in metadata_list:
        print(f"\nVideo: {metadata['video_url']}")
        for key, value in metadata.items():
            if key != 'video_url':
                print(f"{key}: {value:.1f}°")
    
    # Use metadata for improvement vector analysis
    agent = ImprovementVectorProAgent()
    result = agent.analyze_sequence(metadata_list)
    
    return result

if __name__ == "__main__":
    test_videos = [
        "backend/noisy_images/clip_001.mp4",
        "backend/noisy_images/clip_062.mp4",
        "backend/noisy_images/2.mp4"
    ]
    result = get_pro_improvement_vector(test_videos)
    if result['status'] == 'success':
        print("\nImprovement Analysis:")
        print(f"Score: {result['current_score']:.2f}")
        print("\nImprovement Vector:")
        for metric, value in result['improvement_vector'].items():
            print(f"{metric}: {value}")
    else:
        print(f"\nError: {result['message']}")

