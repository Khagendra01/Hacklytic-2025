import streamlit as st
import sys
import os
sys.path.append('../backend')

from preprocessing.noise_masking import isort
from shot_detector import analyze_shot
from agents.quick_analysis import get_coach_feedback
import tempfile

def main():
    st.title("Basketball Shot Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your basketball shot video", type=['mp4', 'mov'])
    
    if uploaded_file is not None:
        # Create a temporary file to save the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        # Create two columns for videos
        col1, col2 = st.columns(2)
        
        with st.spinner('Processing your video...'):
            # Step 1: Noise masking
            isort(video_path)
            
            # Show masked video immediately after processing
            with col1:
                st.subheader("Masked Video")
                st.video('./noisy_images/masked_spot.mp4')
            
            # Step 2: Shot detection and analysis
            analysis_metrics = analyze_shot()
            
            # Step 3: Get coach feedback
            feedback = get_coach_feedback()

            # Show analysis video in second column
            with col2:
                st.subheader("Analysis Visualization")
                st.video('./noisy_images/analysis.mp4')

        # Display results
        st.success('Analysis complete!')
        
        # Display coach recommendations
        st.subheader("Coach Recommendations")
        st.write(feedback['coach_recommendation'])
        
        st.subheader("Detailed Reasoning")
        st.write(feedback['coach_reasoning'])

        # Cleanup
        os.unlink(video_path)
        
if __name__ == "__main__":
    main() 