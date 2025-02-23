import os
import sys
import time
from pathlib import Path
from typing import List, Dict

# Add backend directory to Python path
backend_dir = str(Path(__file__).parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Use relative imports
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

async def process_video_with_timeout(noise_reducer, input_video, unique_output, timeout=30):
    """Process video"""
    try:
        await noise_reducer.process_video(input_video, unique_output)
    except Exception as e:
        print(f"Video processing failed for {input_video}: {str(e)}")
        raise

def fetch_metadata_for_video(input_video, output_dir):
    """Process a single video and return its metrics"""
    try:
        # Create unique output filename for each video
        timestamp = int(time.time() * 1000)
        video_name = Path(input_video).stem
        unique_output = Path('noisy_images/masked_shot.mp4')  # Convert to Path object
        
        print(f"\nProcessing video: {input_video}")
        print(f"Output will be saved to: {unique_output}")
        
        if not os.path.exists(input_video):
            raise FileNotFoundError(f"Input video not found: {input_video}")
        
        try:
            # Process video with noise reduction
            noise_reducer = NoiseReducer()
            asyncio.run(noise_reducer.process_video(str(input_video), str(unique_output)))
            print(f"Created masked video at: {unique_output}")
            
            # Initialize detector and process video
            detector = ShotDetector(str(unique_output))  # Pass video path during initialization
            
            # Get shot metrics
            shot_metrics = detector.shot_metrics
            print("Metrics with mask:", shot_metrics)
            
            # If no metrics detected, try without mask
            if not shot_metrics:
                print("No metrics detected with mask, trying original video...")
                detector = ShotDetector(str(input_video))  # Reinitialize with original video
                shot_metrics = detector.shot_metrics
                print("Metrics without mask:", shot_metrics)
            
            # Get the first frame's metrics if available
            if shot_metrics and len(shot_metrics) > 0:
                metrics = shot_metrics[0]
            else:
                print(f"No metrics detected for video: {input_video}")
                metrics = {
                    'elbow_angle': 0.0,
                    'wrist_angle': 0.0,
                    'shoulder_angle': 0.0,
                    'knee_angle': 0.0,
                    'shot_trajectory': 0.0,
                    'release_height_ratio': 0.0
                }
            
            return metrics
            
        finally:
            # Cleanup temporary masked video
            try:
                if unique_output.exists():  # Now works because unique_output is a Path object
                    unique_output.unlink()
                    print(f"Cleaned up temporary file: {unique_output}")
            except Exception as e:
                print(f"Failed to cleanup temp file: {str(e)}")
                
    except Exception as e:
        print(f"Error processing video {input_video}: {str(e)}")
        return {
            'elbow_angle': 0.0,
            'wrist_angle': 0.0,
            'shoulder_angle': 0.0,
            'knee_angle': 0.0,
            'shot_trajectory': 0.0,
            'release_height_ratio': 0.0
        }

def generate_metadata_from_videos(video_list):
    """Process a list of videos and generate metadata for each"""
    metadata_list = []
    output_dir = Path("noisy_images")
    
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for video_path in video_list:
            try:
                # Process each video
                metrics = fetch_metadata_for_video(video_path, output_dir)
                metadata = {
                    'video_url': video_path,
                    **{k: float(metrics.get(k, 0.0)) for k in [
                        'elbow_angle', 'wrist_angle', 'shoulder_angle',
                        'knee_angle', 'shot_trajectory', 'release_height_ratio'
                    ]}
                }
                
                metadata_list.append(metadata)
                print(f"Successfully processed: {video_path}")
                
            except Exception as e:
                print(f"Failed to process video {video_path}: {str(e)}")
                # Add default metrics for failed video
                metadata_list.append({
                    'video_url': video_path,
                    'elbow_angle': 0.0,
                    'wrist_angle': 0.0,
                    'shoulder_angle': 0.0,
                    'knee_angle': 0.0,
                    'shot_trajectory': 0.0,
                    'release_height_ratio': 0.0
                })
                
    finally:
        # Cleanup temp directory
        try:
            if output_dir.exists():
                for temp_file in output_dir.glob("masked_*.mp4"):
                    temp_file.unlink()
                output_dir.rmdir()
                print("Cleaned up temporary directory")
        except Exception as e:
            print(f"Failed to cleanup temp directory: {str(e)}")
    
    return metadata_list


def get_pro_improvement_vector(test_videos: List) -> Dict:
    try:
        # Generate metadata with timeout
        metadata_list = generate_metadata_from_videos(test_videos)
        
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
    except Exception as e:
        print(f"Error in get_pro_improvement_vector: {str(e)}")
        return {
            'status': 'error',
            'message': str(e)
        }

if __name__ == "__main__":
    try:
        # Use absolute paths relative to the backend directory
        test_videos = [
            "noisy_images/clip_001.mp4",
            "noisy_images/clip_062.mp4",
            "noisy_images/clip_082.mp4"
        ]
        
        print("Starting analysis...")
        result = get_pro_improvement_vector(test_videos)
        
        if result['status'] == 'success':
            print("\nImprovement Analysis:")
            print(f"Score: {result['current_score']:.2f}")
            print("\nImprovement Vector:")
            for metric, value in result['improvement_vector'].items():
                print(f"{metric}: {value}")
        else:
            print(f"\nError: {result['message']}")
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        # Ensure cleanup happens even on keyboard interrupt
        if 'output_dir' in locals():
            try:
                for temp_file in Path("temp_processed_videos").glob("masked_*.mp4"):
                    temp_file.unlink()
                Path("temp_processed_videos").rmdir()
            except Exception as e:
                print(f"Cleanup failed: {str(e)}")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")

