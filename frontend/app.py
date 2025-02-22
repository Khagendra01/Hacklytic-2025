import os
import sys
import time
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.append(str(backend_path))

from preprocessing.noise_masking import NoiseReducer
from shot_detector import ShotDetector
from agents.quick_analysis_agent import QuickAnalysisAgent

def create_temp_dirs():
    """Create temporary directories if they don't exist"""
    os.makedirs("backend/noisy_images", exist_ok=True)
    os.makedirs("temp_uploads", exist_ok=True)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory"""
    file_path = os.path.join("temp_uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def display_video(video_path, width=None):
    """Display video in Streamlit"""
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
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
        # Save uploaded file
        input_path = save_uploaded_file(uploaded_file)
        
        # Initialize progress tracking
        progress_text = "Operation in progress. Please wait..."
        progress_bar = st.progress(0, text=progress_text)
        
        try:
            # Create two columns for video display
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Video")
                display_video(input_path)
            
            # Process video with noise masking
            noise_reducer = NoiseReducer()
            masked_video_path = "backend/noisy_images/masked_shot.mp4"
            
            with st.spinner("Processing video..."):
                # Run noise reduction
                import asyncio
                asyncio.run(noise_reducer.process_video(input_path, masked_video_path))
                progress_bar.progress(33, text="Noise masking complete...")
                
                with col2:
                    st.subheader("Masked Video")
                    display_video(masked_video_path)
                
                # Run shot detection
                detector = ShotDetector()
                progress_bar.progress(66, text="Shot detection complete...")
                
                # Run quick analysis
                analyzer = QuickAnalysisAgent()
                analysis = analyzer.analyze_form(detector.shot_metrics)
                progress_bar.progress(100, text="Analysis complete!")
                
                # Display analysis results
                st.subheader("Analysis Results")
                
                # Create expandable sections for recommendations and reasoning
                with st.expander("Coach Recommendations", expanded=True):
                    recommendations = analysis["coach_recommendations"].split("||")
                    for rec in recommendations:
                        st.write(rec.strip())
                
                with st.expander("Detailed Analysis"):
                    reasoning = analysis["coach_reasoning"].split("||")
                    for reason in reasoning:
                        st.write(reason.strip())
                
                # Display shot metrics in a table
                if detector.shot_metrics:
                    st.subheader("Shot Metrics")
                    metrics_df = pd.DataFrame(detector.shot_metrics)
                    st.dataframe(metrics_df)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
        finally:
            # Cleanup
            if os.path.exists(input_path):
                os.remove(input_path)
            progress_bar.empty()

if __name__ == "__main__":
    main() 