import asyncio
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

# Add this workaround for PyTorch + Python 3.12 compatibility
import torch._dynamo.config

torch._dynamo.config.suppress_errors = True

import cv2
import pandas as pd
from PIL import Image

import streamlit as st

# Get absolute paths
ROOT_DIR = Path(__file__).parent.parent.absolute()

BACKEND_DIR = ROOT_DIR / "backend"
TEMP_DIR = BACKEND_DIR / "temp_uploads"
NOISY_IMAGES_DIR = BACKEND_DIR / "noisy_images"
print(BACKEND_DIR)
# Add backend to path
sys.path.append(str(BACKEND_DIR))

from agents.quick_analysis_agent import QuickAnalysisAgent
from agents.video_reconstruction_agent import VideoReconstructionAgent
from preprocessing.noise_masking import NoiseReducer
from shot_detector import ShotDetector


def create_temp_dirs():
    """Create temporary directories if they don't exist"""
    NOISY_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory"""
    file_path = TEMP_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)

def display_video(video_bytes, width=None):
    """Display video using Streamlit's native video display"""
    if width:
        st.video(video_bytes, width=width)
    else:
        st.video(video_bytes)

def main():
    st.title("Basketball Shot Analysis")
    
    # Create necessary directories
    create_temp_dirs()
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your basketball shot video", type=['mp4', 'mov', 'avi'])
    
    if uploaded_file:
        try:
            # Display original video directly from bytes
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Video")
                display_video(uploaded_file.getvalue())
            
            # Process video with noise masking
            progress_text = "Operation in progress. Please wait..."
            progress_bar = st.progress(0, text=progress_text)
            
            with st.spinner("Processing video..."):
                # Create temporary BytesIO objects for processing
                input_bytes = uploaded_file.getvalue()
                
                # Save to temp file for processing (Streamlit will clean this up)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
                    tmp_input.write(input_bytes)
                    tmp_input_path = tmp_input.name
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_output:
                    tmp_output_path = tmp_output.name
                
                try:
                    # Process video
                    noise_reducer = NoiseReducer()
                    noise_reducer.process_video_sync(tmp_input_path, tmp_output_path)
                    progress_bar.progress(33, text="Noise masking complete...")
                    
                    # Read processed video
                    with open(tmp_output_path, 'rb') as f:
                        processed_video = f.read()
                        
                    with col2:
                        st.subheader("Processed Video")
                        display_video(processed_video)
                    
                    # Run shot detection with fallback
                    detector = ShotDetector()
                    shot_metrics = detector.process_video(tmp_input_path)
                    progress_bar.progress(66, text="Shot detection complete...")
                    
                    # Add warning but continue processing
                    if not shot_metrics:
                        st.warning("No shots detected automatically. Proceeding with basic form analysis.")
                        # Create a default metric set
                        shot_metrics = [{
                            'elbow_angle': 0,
                            'wrist_angle': 0,
                            'shoulder_angle': 0,
                            'knee_angle': 0,
                            'shot_trajectory': 0,
                            'release_height_ratio': 0,
                            'player_height_pixels': 0,
                            'torso_size_pixels': 0,
                            'torso_ratio': 0,
                            'release_height': 0,
                            'ideal_ranges': {
                                'elbow_angle': (165, 175),
                                'wrist_angle': (70, 90),
                                'shoulder_angle': (90, 110),
                                'knee_angle': (140, 170),
                                'shot_trajectory': (45, 55),
                                'release_height_ratio': (1.8, 2.3),
                                'torso_ratio': (0.3, 0.4)
                            }
                        }]
                    
                    # Run analysis with either detected or default metrics
                    analyzer = QuickAnalysisAgent()
                    analysis = analyzer.analyze_form(shot_metrics)
                    progress_bar.progress(100, text="Analysis complete!")
                    
                    # Display results
                    st.subheader("Analysis Results")
                    
                    with st.expander("Coach Recommendations", expanded=True):
                        if "coach_recommendations" in analysis:
                            recommendations = analysis["coach_recommendations"].split("||")
                            for rec in recommendations:
                                st.write(rec.strip())
                        else:
                            st.write("Unable to generate specific recommendations. Try uploading a clearer video of your shot.")
                    
                    with st.expander("Detailed Analysis"):
                        if "coach_reasoning" in analysis:
                            reasoning = analysis["coach_reasoning"].split("||")
                            for reason in reasoning:
                                st.write(reason.strip())
                        else:
                            st.write("Detailed analysis not available. Ensure the video shows a complete basketball shot.")
                    
                    # Display metrics if available
                    if shot_metrics:
                        st.subheader("Shot Metrics")
                        metrics_df = pd.DataFrame(shot_metrics)
                        st.dataframe(metrics_df)
                    
                    # Always attempt video reconstruction
                    reconstructor = VideoReconstructionAgent()
                    ideal_video_path = TEMP_DIR / "ideal_form.mp4"
                    
                    try:
                        result = reconstructor.process_video(tmp_input_path, str(ideal_video_path))
                        
                        # Display ideal form video if available
                        if Path(result['video_path']).exists():
                            with open(result['video_path'], 'rb') as f:
                                ideal_video = f.read()
                            
                            st.subheader("Ideal Form")
                            display_video(ideal_video)
                        
                        # Display feedback
                        st.subheader("Form Adjustments")
                        if result['feedback']['summary']:
                            for adjustment in result['feedback']['summary']:
                                st.write(adjustment)
                        else:
                            st.write("No specific form adjustments detected. Try recording from a different angle.")
                        
                    except Exception as e:
                        st.warning("Could not generate ideal form visualization. Please ensure the video shows a clear view of the shot.")
                        st.error(f"Technical details: {str(e)}")
                
                except Exception as e:
                    st.error(f"Error in video reconstruction: {str(e)}")
                    st.error(traceback.format_exc())
                
                finally:
                    # Clean up temporary files
                    if os.path.exists(tmp_input_path):
                        os.unlink(tmp_input_path)
                    if os.path.exists(tmp_output_path):
                        os.unlink(tmp_output_path)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            # Print full error details for debugging
            st.error(traceback.format_exc())
        
        finally:
            progress_bar.empty()

if __name__ == "__main__":
    main() 