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
from langchain_openai import ChatOpenAI
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
    def __init__(self):
        self.video_path = 'backend/noisy_images/analysis.mp4'
        self.model = genai.GenerativeModel("gemini-1.5-pro")
        
    def analyze_video(self) -> Dict[str, str]:
        """Analyze the video content using Gemini Vision."""
        print("Uploading video file...")
        # Upload the video file using the Files API
        video_file = genai.upload_file(self.video_path)
        print(f"Completed upload: {video_file.name}")

        # Wait for video processing
        while video_file.state.name == "PROCESSING":
            print('.', end='', flush=True)
            time.sleep(5)  # Increased sleep time to reduce API calls
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.error}")

        print('\nVideo processing complete')

        prompt = """
        You are an expert basketball free throw coach. Analyze this free throw shot video.
        Focus on:
        1. Starting position and setup
        2. Body alignment through the shot
        3. Shot mechanics (elbow, wrist, shoulder positioning)
        4. Release point and follow-through
        5. Overall flow and rhythm

        Provide specific feedback on:
        1. What the player is doing well
        2. Key areas for improvement
        3. Specific form corrections needed
        4. Tips for better consistency

        Also provide a detailed breakdown with timestamps of key moments in the shot.
        """

        # Generate analysis using the video file
        response = self.model.generate_content([
            video_file,
            prompt
        ])
        
        # Clean up the uploaded file
        video_file.delete()
        
        visual_analysis = response.text.split('\n\n')
        recommendations = [line for line in visual_analysis if line.startswith(('Good:', 'Improve:', 'Tip:'))]
        reasoning = [line for line in visual_analysis if not line.startswith(('Good:', 'Improve:', 'Tip:'))]
        
        return {
            "visual_recommendations": " | ".join(recommendations),
            "visual_reasoning": " | ".join(reasoning)
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

def initialize_agent():
    """Initialize the quick analysis agent using Gemini."""
    model = genai.GenerativeModel('gemini-1.5-pro')
    analyzer = QuickAnalysisAgent()
    
    def analyze_shot(metrics_data: list) -> Dict[str, str]:
        # Prepare the prompt with metrics data
        prompt = (
            "As an expert free throw coach, analyze this player's shooting form metrics:\n\n"
            f"{json.dumps(metrics_data, indent=2)}\n\n"
            "Focus on:\n"
            "1. The most critical aspects that need immediate attention\n"
            "2. Specific drills to improve these areas\n"
            "3. Positive aspects of their form to build upon\n"
            "4. Mental tips for consistent free throw shooting\n\n"
            "Provide your response in strict JSON format like this:\n"
            '{\n'
            '    "coach_recommendations": "Write your key points for improvement here",\n'
            '    "coach_reasoning": "Write your detailed analysis and explanation here"\n'
            '}\n'
            'Important: Response must be valid JSON only, no other text.'
        )

        # Get analysis from Gemini
        response = model.generate_content(prompt)
        
        # Print response for debugging
        print("Raw response from Gemini:")
        print(response.text)
        
        try:
            # Try to parse as JSON
            metrics_analysis = json.loads(response.text)
        except json.JSONDecodeError:
            # If JSON parsing fails, create a structured response from the text
            metrics_analysis = {
                "coach_recommendations": "Analysis format error",
                "coach_reasoning": response.text
            }
        
        video_analysis = analyzer.analyze_form(metrics_data)
        
        return {
            "coach_recommendations": (
                f"Video Analysis: {video_analysis['coach_recommendations']}\n"
                f"Metrics Analysis: {metrics_analysis['coach_recommendations']}"
            ),
            "coach_reasoning": (
                f"Video Breakdown: {video_analysis['coach_reasoning']}\n"
                f"Metrics Breakdown: {metrics_analysis['coach_reasoning']}"
            )
        }

    return analyze_shot

def main():
    # Test data
    test_metrics = [{
        'elbow_angle': 176.23948929424307,
        'wrist_angle': 179.57559458063852,
        'shoulder_angle': 11.791874720340788,
        'knee_angle': 174.5554432396471,
        'shot_trajectory': -53.27286562892127,
        'release_height_ratio': 2.9242424242424243,
        'player_height_pixels': 115,
        'torso_size_pixels': 49,
        'torso_ratio': 0.4260869565217391,
        'release_height': 242,
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

    # Initialize and run the agent
    analyze_shot = initialize_agent()
    result = analyze_shot(test_metrics)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
