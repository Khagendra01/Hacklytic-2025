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
        self.video_path = 'backend/noisy_images/analysis.mp4'  # Default analysis output path
        self.model = genai.GenerativeModel("gemini-1.5-pro")
        
    def set_video(self, video_url: str):
        """Update the video URL after initialization"""
        self.video_url = video_url
        
    def analyze_video(self) -> Dict[str, str]:
        """Analyze the video content using Gemini Vision."""
        try:
            if not self.video_url or not os.path.exists(self.video_url):
                return {
                    "visual_recommendations": "Video file not found",
                    "visual_reasoning": f"Unable to analyze video: File not found at {self.video_url}"
                }

            print(f"Analyzing video: {self.video_url}")
            
            # Read video file as binary
            with open(self.video_url, 'rb') as f:
                video_bytes = f.read()
            
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

            prompt = """
            You are an expert basketball free throw coach. Analyze this free throw shot video.
            Use the metrics data to:
            1,
            Create a comparison of current metrics compared to ideal ranges such as the example below and suggest improvements to the user:
            Ex. Elbow Angle (172.15°) ✅ Within range (165°–175°)
                Knee Angle (177.20°) ❌ Exceeds range (140°–170°)
                Release Height Ratio (1.78) ❌ Below range (1.8–2.3)
                Shot Trajectory (-80.19°) ❌ Way off from range (45°–55°)
                Shoulder Angle (144.21°) ❌ Exceeds range (90°–110°)
                Torso Ratio (0.365) ✅ Within range (0.3–0.4)
                Wrist Angle (171.32°) ❌ Exceeds range (70°–90°)

            2,
            Dive into, Deviation from Ideal Values
            Quantifying how much the player deviates from the ideal range:

            Ex. Knee Angle: +7.20° (Overextended knees)
            Release Height Ratio: -0.02 (Slightly lower than ideal)
            Shot Trajectory: -125.19° (Very far from the ideal range)
            Shoulder Angle: +34.21° (Overextended shoulder)
            Wrist Angle: +81.32° (Overextended wrist)

            3, summarise by collating suggestions as seen,
            Ex. Overextended Knee & Shoulder Angles → Possible rigid form; may cause inefficiency in power transfer.
            Wrist Overextension → Could lead to inconsistent spin and trajectory.
            Shot Trajectory (-80°) → This is likely a miscalculation or a major issue since the ideal range is 45°–55°. Check if the data source for this value is correct.
            Release Height Slightly Low → This could be a timing issue or a lower-than-optimal release point.

            NOTE: DO NOT ADD ANY ADDITIONAL COMMENTS ABOUT VIDEO QUALITY, CAMERA ANGLE, OR ACCURACY OF DATA.
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

            visual_analysis = response.text.split('\n\n')
            recommendations = [line for line in visual_analysis if line.startswith(('Good:', 'Improve:', 'Tip:'))]
            reasoning = [line for line in visual_analysis if not line.startswith(('Good:', 'Improve:', 'Tip:'))]
            
            return {
                "visual_recommendations": " | ".join(recommendations) if recommendations else "",
                "visual_reasoning": " | ".join(reasoning) if reasoning else ""
            }

        except Exception as e:
            print(f"Error in analyze_video: {str(e)}")
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

        metrics = ShootingMetrics.from_dict(metrics_list[0])  # Analyze first shot
        
        # Get video analysis
        video_analysis = self.analyze_video()
        
        # Generate analysis based on metrics
        recommendations = []
        reasoning = []

        # Analyze elbow angle
        if metrics.elbow_angle > metrics.ideal_ranges['elbow_angle'][1]:
            recommendations.append("Keep your elbow slightly bent at release")
            reasoning.append(f"Your elbow angle ({metrics.elbow_angle:.1f}°) is too straight. " 
                           f"Ideal range is {metrics.ideal_ranges['elbow_angle'][0]}-{metrics.ideal_ranges['elbow_angle'][1]}°")

        # Analyze shoulder angle
        if metrics.shoulder_angle < metrics.ideal_ranges['shoulder_angle'][0]:
            recommendations.append("Raise your shooting pocket higher")
            reasoning.append(f"Your shoulder angle ({metrics.shoulder_angle:.1f}°) is too low. "
                           f"Ideal range is {metrics.ideal_ranges['shoulder_angle'][0]}-{metrics.ideal_ranges['shoulder_angle'][1]}°")

        # Analyze shot trajectory
        if abs(metrics.shot_trajectory) > metrics.ideal_ranges['shot_trajectory'][1]:
            recommendations.append("Adjust your shot arc to be more balanced")
            reasoning.append(f"Your shot trajectory ({abs(metrics.shot_trajectory):.1f}°) is off. "
                           f"Ideal range is {metrics.ideal_ranges['shot_trajectory'][0]}-{metrics.ideal_ranges['shot_trajectory'][1]}°")

        # Analyze release height
        if metrics.release_height_ratio > metrics.ideal_ranges['release_height_ratio'][1]:
            recommendations.append("Lower your release point slightly")
            reasoning.append(f"Your release height ratio ({metrics.release_height_ratio:.2f}) is too high. "
                           f"Ideal range is {metrics.ideal_ranges['release_height_ratio'][0]}-{metrics.ideal_ranges['release_height_ratio'][1]}")

        if not recommendations:
            recommendations.append("Your shooting form looks good! Keep practicing for consistency.")
            reasoning.append("All metrics are within ideal ranges.")

        # Combine metric and video analysis with more weight on visual analysis
        return {
            "coach_recommendations": video_analysis["visual_recommendations"] + " || Metrics-based tips: " + " | ".join(recommendations),
            "coach_reasoning": video_analysis["visual_reasoning"] + " || Metric analysis: " + " | ".join(reasoning)
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
    # Example usage
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
            "elbow_angle": [
                165,
                175
            ],
            "wrist_angle": [
                70,
                90
            ],
            "shoulder_angle": [
                90,
                110
            ],
            "knee_angle": [
                140,
                170
            ],
            "shot_trajectory": [
                45,
                55
            ],
            "release_height_ratio": [
                1.8,
                2.3
            ],
            "torso_ratio": [
                0.3,
                0.4
            ]
        }
    }]
    
    video_path = "noisy_images/clip_001.mp4"
    result = get_quick_analysis(test_metrics, video_path)
    print('RESULT FETCHED', result)