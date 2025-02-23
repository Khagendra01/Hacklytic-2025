import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict

import cv2
import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core import exceptions
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

@dataclass
class ShootingMetrics:
    elbow_angle: float
    wrist_angle: float
    shoulder_angle: float
    knee_angle: float
    shot_trajectory: float
    release_height_ratio: float
    player_height_pixels: float
    torso_size_pixels: float
    torso_ratio: float
    release_height: float
    ideal_ranges: Dict[str, tuple]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShootingMetrics':
        return cls(
            elbow_angle=float(data.get('elbow_angle', 0)),
            wrist_angle=float(data.get('wrist_angle', 0)),
            shoulder_angle=float(data.get('shoulder_angle', 0)),
            knee_angle=float(data.get('knee_angle', 0)),
            shot_trajectory=float(data.get('shot_trajectory', 0)),
            release_height_ratio=float(data.get('release_height_ratio', 0)),
            player_height_pixels=float(data.get('player_height_pixels', 0)),
            torso_size_pixels=float(data.get('torso_size_pixels', 0)),
            torso_ratio=float(data.get('torso_ratio', 0)),
            release_height=float(data.get('release_height', 0)),
            ideal_ranges=data.get('ideal_ranges', {})
        )

class QuickAnalysisAgent:
    def __init__(self, video_url: str = None):
        """
        Initialize the agent with a video URL
        Args:
            video_url: Path to the video file to analyze
        """
        self.video_url = video_url
        
        # Configure Gemini
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel("gemini-1.5-pro")
        
        print(f"Agent initialized with video: {video_url}")
        
    def set_video(self, video_url: str):
        """Update the video URL after initialization"""
        self.video_url = video_url
        
    def analyze_video(self, metrics_data: Dict) -> Dict[str, str]:
        """Analyze the video content using Gemini Vision."""
        try:
            if not self.video_url or not os.path.exists(self.video_url):
                raise FileNotFoundError(f"Video file not found at: {self.video_url}")

            print(f"Starting analysis of video: {self.video_url}")
            print(f"Using metrics: {metrics_data}")
            
            # Read video file as binary
            with open(self.video_url, 'rb') as f:
                video_bytes = f.read()
                print(f"Successfully read video file ({len(video_bytes)} bytes)")
            
            # Format metrics data for prompt
            metrics_str = "\n".join([
                f"{key}: {value:.2f}°" if isinstance(value, (int, float)) else f"{key}: {value}"
                for key, value in metrics_data.items()
                if key != 'ideal_ranges' and not isinstance(value, (list, dict))
            ])
            
            # Format ideal ranges
            ideal_ranges = metrics_data.get('ideal_ranges', {})
            ranges_str = "\n".join([
                f"{key} ideal range: {range_[0]}-{range_[1]}°"
                for key, range_ in ideal_ranges.items()
            ])

            print("\nFormatted metrics for analysis:")
            print(metrics_str)
            print("\nIdeal ranges:")
            print(ranges_str)

            # Create generation config
            generation_config = {
                "temperature": 0.4,
                "top_p": 1,
                "top_k": 32,
                "max_output_tokens": 2048,
                
            }

            # Create safety settings
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]

            prompt = f"""
            You are an expert basketball free throw coach. Analyze this free throw shot video and the provided metrics.

            Current Metrics:
            {metrics_str}

            Ideal Ranges:
            {ranges_str}

            Please provide:

            1. Metric Analysis:
            - Compare each metric to its ideal range
            - Use ✅ for metrics within range and ❌ for those outside
            - Format as: Metric (Current°) [✅/❌] Range (min°-max°)

            2. Deviation Analysis:
            - Calculate how far each metric is from ideal range
            - Show + for over and - for under
            - Focus on significant deviations

            3. Form Improvement Summary:
            - Group related issues (e.g., connected joint angles)
            - Prioritize most critical adjustments
            - Provide specific, actionable feedback
            - Explain impact on shot effectiveness

            NOTE: Focus only on the metrics and visual form. Do not comment on video quality or data accuracy.
            """

            # Generate content with video
            response = self.model.generate_content(
                contents=[
                    {
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {
                                "mime_type": "video/mp4",
                                "data": base64.b64encode(video_bytes).decode('utf-8')
                            }}
                        ]
                    }
                ],
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            if response.prompt_feedback:
                print("Prompt feedback:", response.prompt_feedback)

            # Get the response text
            response_text = response.text
            
            # Split into main sections using the numbered headers
            sections = {}
            current_section = None
            current_text = []
            
            for line in response_text.split('\n'):
                # Check for main section headers
                if '1. Metric Analysis:' in line:
                    current_section = 'metric_analysis'
                    current_text = []
                elif '2. Deviation Analysis:' in line:
                    if current_section:
                        sections[current_section] = '\n'.join(current_text)
                    current_section = 'deviation_analysis'
                    current_text = []
                elif '3. Form Improvement Summary:' in line:
                    if current_section:
                        sections[current_section] = '\n'.join(current_text)
                    current_section = 'improvement_summary'
                    current_text = []
                elif 'Prioritized Adjustments:' in line:
                    if current_section:
                        sections[current_section] = '\n'.join(current_text)
                    current_section = 'prioritized_adjustments'
                    current_text = []
                elif line.strip() and current_section:  # Add non-empty lines to current section
                    current_text.append(line.strip())
            
            # Add the last section
            if current_section and current_text:
                sections[current_section] = '\n'.join(current_text)

            # Construct the recommendations and reasoning
            recommendations = []
            if 'improvement_summary' in sections:
                recommendations.append("Form Improvement Summary:")
                recommendations.append(sections['improvement_summary'])
            if 'prioritized_adjustments' in sections:
                recommendations.append("\nPrioritized Adjustments:")
                recommendations.append(sections['prioritized_adjustments'])

            reasoning = []
            if 'metric_analysis' in sections:
                reasoning.append("Metric Analysis:")
                reasoning.append(sections['metric_analysis'])
            if 'deviation_analysis' in sections:
                reasoning.append("\nDeviation Analysis:")
                reasoning.append(sections['deviation_analysis'])

            return {
                "visual_recommendations": '\n'.join(recommendations),
                "visual_reasoning": '\n'.join(reasoning)
            }

        except Exception as e:
            print(f"Error in analyze_video: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "visual_recommendations": "Error analyzing video",
                "visual_reasoning": f"An error occurred: {str(e)}"
            }

    def analyze_form(self, metrics_list: list) -> Dict[str, str]:
        if not metrics_list:
            return {
                "coach_recommendations": "No metrics data available for analysis.",
                "coach_reasoning": "Unable to analyze shooting form without metrics data."
            }

        metrics = metrics_list[0]  # Get first frame's metrics
        
        # Get video analysis with metrics
        video_analysis = self.analyze_video(metrics)
        
        return {
            "coach_recommendations": video_analysis["visual_recommendations"],
            "coach_reasoning": video_analysis["visual_reasoning"]
        }

def initialize_agent(video_url: str = None):
    """Initialize the quick analysis agent using Gemini."""
    model = genai.GenerativeModel('gemini-1.5-pro')
    analyzer = QuickAnalysisAgent(video_url)
    
    def analyze_shot(metrics_data: list) -> Dict[str, str]:
        if video_url:
            analyzer.set_video(video_url)
        return analyzer.analyze_form(metrics_data)

    return analyze_shot

def get_quick_analysis(metrics_data: list, video_url: str = None) -> Dict[str, str]:
    """
    Get quick analysis for metrics and video
    Args:
        metrics_data: List of metrics dictionaries
        video_url: Optional path to video file
    """
    analyze_shot = initialize_agent(video_url)
    result = analyze_shot(metrics_data)
    return result

if __name__ == "__main__":
    import os
    from pathlib import Path

    # Get the backend directory path
    backend_dir = Path(__file__).parent.parent
    
    # Example usage with proper paths
    test_metrics = [{
        "elbow_angle": 172.15237249531512,
        "wrist_angle": 171.32183111475928,
        "shoulder_angle": 144.21102654081665,
        "knee_angle": 177.204574709685,
        "shot_trajectory": -80.18521332794774,
        "release_height_ratio": 1.776470588235294,
        "player_height_pixels": 189,
        "torso_size_pixels": 69,
        "torso_ratio": 0.36507936507936506,
        "release_height": 255,
        "ideal_ranges": {
            "elbow_angle": [165, 175],
            "wrist_angle": [70, 90],
            "shoulder_angle": [90, 110],
            "knee_angle": [140, 170],
            "shot_trajectory": [45, 55],
            "release_height_ratio": [1.8, 2.3],
            "torso_ratio": [0.3, 0.4]
        }
    }]
    
    # Construct proper path to video
    video_path = str(backend_dir / "noisy_images" / "clip_001.mp4")
    
    print(f"Initializing analysis for video: {video_path}")
    print(f"With metrics: {test_metrics}")
    
    result = get_quick_analysis(test_metrics, video_path)
    
    print("\n" + "="*80)
    print("COACH RECOMMENDATIONS:")
    print("="*80)
    print(result["coach_recommendations"])
    
    print("\n" + "="*80)
    print("COACH REASONING:")
    print("="*80)
    print(result["coach_reasoning"])
   