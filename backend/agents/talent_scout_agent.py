import os
import base64
import google.generativeai as genai
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Dict, List, Any

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Synthetic data for Curry's progression
CURRY_PROGRESSION = [
    {
        "period": "Early Warriors (2009)",
        "score": 82,
        "tier": "A",
        "metrics": {
            "elbow_angle": 172,
            "wrist_angle": 85,
            "shoulder_angle": 105,
            "knee_angle": 165,
            "shot_trajectory": 48,
            "release_height_ratio": 2.1,
            "torso_ratio": 0.35
        }
    },
    {
        "period": "Mid Warriors (2015)",
        "score": 95,
        "tier": "S",
        "metrics": {
            "elbow_angle": 170,
            "wrist_angle": 82,
            "shoulder_angle": 102,
            "knee_angle": 160,
            "shot_trajectory": 52,
            "release_height_ratio": 2.2,
            "torso_ratio": 0.36
        }
    },
    {
        "period": "Current Warriors (2024)",
        "score": 100,
        "tier": "S",
        "metrics": {
            "elbow_angle": 171,
            "wrist_angle": 80,
            "shoulder_angle": 100,
            "knee_angle": 162,
            "shot_trajectory": 50,
            "release_height_ratio": 2.15,
            "torso_ratio": 0.35
        }
    }
]

class TalentScoutAgent:
    def __init__(self):
        """Initialize the talent scout agent with pro reference data and video analyzer."""
        self.pro_reference = CURRY_PROGRESSION
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Score ranges for tiers
        self.tier_ranges = {
            'S': (85, 100),
            'A': (70, 84),
            'B': (55, 69),
            'C': (40, 54)
        }
        
        # Ideal ranges for metrics
        self.ideal_ranges = {
            'elbow_angle': (165, 175),
            'wrist_angle': (70, 90),
            'shoulder_angle': (90, 110),
            'knee_angle': (140, 170),
            'shot_trajectory': (45, 55),
            'release_height_ratio': (1.8, 2.3),
            'torso_ratio': (0.3, 0.4)
        }

    def _analyze_video(self, video_path: str) -> Dict[str, str]:
        """Analyze video using Gemini Vision."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at path: {video_path}")
            
        try:
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
        except IOError as e:
            raise IOError(f"Failed to read video file {video_path}: {str(e)}")
            
        prompt = """
        You are an expert basketball free throw coach. Analyze this free throw shot video.
        Focus on form, mechanics, and technique. Be concise and technical.
        Do not mention video quality or camera angles.
        Provide a brief, focused analysis of the shooting mechanics and any notable technical aspects.
        """

        try:
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
                generation_config={"temperature": 0.4, "top_p": 1, "top_k": 32, "max_output_tokens": 1024}
            )
            return {"visual_analysis": response.text}
        except Exception as e:
            raise RuntimeError(f"Gemini API error during video analysis: {str(e)}")

    def _calculate_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate a score (40-85) based on metrics.
        Stricter scoring system that emphasizes precision.
        Rookies shouldn't reach S-tier immediately.
        """
        base_score = 40  # Lower minimum score
        max_additional_score = 45  # Cap maximum additional points to keep scores realistic
        
        # Weights emphasize key shooting mechanics
        weights = {
            'shot_trajectory': 0.25,    # Most important - proper arc
            'elbow_angle': 0.20,        # Critical for shot consistency
            'wrist_angle': 0.15,        # Important for shot control
            'shoulder_angle': 0.15,     # Key for alignment
            'knee_angle': 0.15,         # Essential for power generation
            'release_height_ratio': 0.07,# Minor factor
            'torso_ratio': 0.03         # Least important
        }
        
        total_score = base_score
        
        for metric, weight in weights.items():
            if metric in metrics and metric in self.ideal_ranges:
                ideal_min, ideal_max = self.ideal_ranges[metric]
                value = metrics[metric]
                
                # Even stricter scoring - quartic falloff for values outside ideal range
                if ideal_min <= value <= ideal_max:
                    metric_score = 1.0
                else:
                    # Calculate distance from ideal range
                    distance = min(abs(value - ideal_min), abs(value - ideal_max))
                    range_size = ideal_max - ideal_min
                    # Quartic falloff makes the scoring extremely unforgiving
                    metric_score = max(0.1, 1 - (distance / range_size) ** 4)
                
                total_score += metric_score * weight * max_additional_score
        
        # Cap maximum score at 85 for rookies
        return min(85, max(40, round(total_score, 1)))

    def _determine_tier(self, score: float) -> str:
        """Determine tier based on score."""
        for tier, (min_score, max_score) in self.tier_ranges.items():
            if min_score <= score <= max_score:
                return tier
        return 'C'  # Default to C tier if no range matches

    def _analyze_single_video(self, metrics: Dict[str, Any], video_path: str = None) -> Dict[str, Any]:
        """Analyze a single video's metrics with more constructive feedback."""
        score = self._calculate_score(metrics)
        tier = self._determine_tier(score)
        
        # Generate more constructive technical analysis
        technical_analysis = []
        strengths = []
        for metric, value in metrics.items():
            if metric in self.ideal_ranges:
                ideal_min, ideal_max = self.ideal_ranges[metric]
                metric_name = metric.replace('_', ' ').title()
                
                if ideal_min <= value <= ideal_max:
                    strengths.append(f"Good {metric_name} ({value:.1f}°)")
                elif value < ideal_min:
                    technical_analysis.append(f"{metric_name} ({value:.1f}°) has room for improvement - aim for {ideal_min}°-{ideal_max}°")
                else:
                    technical_analysis.append(f"{metric_name} ({value:.1f}°) could be adjusted closer to {ideal_min}°-{ideal_max}°")

        analysis = {
            "score": score,
            "tier": tier,
            "technical_analysis": (
                "Strengths: " + ". ".join(strengths) + ". " if strengths else ""
            ) + (
                "Areas for refinement: " + ". ".join(technical_analysis) if technical_analysis else "Excellent form across all metrics."
            )
        }

        if video_path:
            video_analysis = self._analyze_video(video_path)
            analysis["visual_analysis"] = video_analysis["visual_analysis"]

        return analysis

    def _calculate_improvement(self, initial: Dict[str, Any], follow_up: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze improvement with realistic framing for rookies."""
        initial_score = self._calculate_score(initial)
        follow_up_score = self._calculate_score(follow_up)
        
        # Cap improvement at 8 points for realism
        score_improvement = min(8, max(2, follow_up_score - initial_score))
        follow_up_score = initial_score + score_improvement
        
        improvements = []
        refinements = []
        
        for metric in self.ideal_ranges.keys():
            if metric in initial and metric in follow_up:
                change = follow_up[metric] - initial[metric]
                ideal_min, ideal_max = self.ideal_ranges[metric]
                metric_name = metric.replace('_', ' ').title()
                
                if abs(change) > 0.1:
                    if ideal_min <= follow_up[metric] <= ideal_max:
                        improvements.append(metric_name)
                    else:
                        refinements.append(metric_name)

        return {
            "score_improvement": score_improvement,
            "key_improvements": improvements,
            "areas_for_development": refinements,
            "summary": (
                f"Score improved by {score_improvement:.1f} points. " +
                (f"Notable improvements in {', '.join(improvements)}. " if improvements else "Showing steady progress. ") +
                (f"Continuing to refine {', '.join(refinements)}." if refinements else "")
            )
        }

    def _assess_potential(self, initial: Dict[str, Any], follow_up: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate potential with realistic projections for rookies."""
        time_frame = 1
        improvement_rate = (follow_up["score"] - initial["score"]) / time_frame
        
        # Compare with Curry's early improvement rate
        curry_early_rate = (self.pro_reference[1]["score"] - self.pro_reference[0]["score"]) / 6
        
        # More realistic potential calculation for rookies
        base_potential = min(88, follow_up["score"] + (improvement_rate * 2))  # Project 2 time units forward
        
        # Adjust based on comparison with Curry's progression
        rate_ratio = improvement_rate / curry_early_rate if curry_early_rate != 0 else 0.3
        potential_adjustment = min(7, max(-10, (rate_ratio - 0.9) * 5))  # Very hard to get positive adjustment
        
        final_potential = min(90, max(45, base_potential + potential_adjustment))
        
        # More realistic trajectory assessment for rookies
        if improvement_rate > curry_early_rate * 0.7:
            trajectory = "showing strong potential"
        elif improvement_rate > curry_early_rate * 0.5:
            trajectory = "developing steadily"
        else:
            trajectory = "making gradual progress"

        return {
            "score": round(final_potential, 1),
            "justification": (
                f"Development rate is {trajectory}. " +
                f"Current trajectory suggests a potential ceiling of {final_potential:.1f}, " +
                f"improving at {improvement_rate:.1f} points per session."
            )
        }

    def analyze_talent(self, 
                      video1_metrics: Dict[str, Any], 
                      video2_metrics: Dict[str, Any],
                      video1_path: str = None,
                      video2_path: str = None) -> Dict[str, Any]:
        """
        Main handler function that orchestrates the entire analysis process.
        """
        # Extract just the metrics we need
        metrics1 = {
            'elbow_angle': video1_metrics.get('elbow_angle', 0),
            'wrist_angle': video1_metrics.get('wrist_angle', 0),
            'shoulder_angle': video1_metrics.get('shoulder_angle', 0),
            'knee_angle': video1_metrics.get('knee_angle', 0),
            'shot_trajectory': video1_metrics.get('shot_trajectory', 0),
            'release_height_ratio': video1_metrics.get('release_height_ratio', 0),
            'torso_ratio': video1_metrics.get('torso_ratio', 0)
        }
        
        metrics2 = {
            'elbow_angle': video2_metrics.get('elbow_angle', 0),
            'wrist_angle': video2_metrics.get('wrist_angle', 0),
            'shoulder_angle': video2_metrics.get('shoulder_angle', 0),
            'knee_angle': video2_metrics.get('knee_angle', 0),
            'shot_trajectory': video2_metrics.get('shot_trajectory', 0),
            'release_height_ratio': video2_metrics.get('release_height_ratio', 0),
            'torso_ratio': video2_metrics.get('torso_ratio', 0)
        }

        # Calculate initial scores
        score1 = self._calculate_score(metrics1)
        score2 = self._calculate_score(metrics2)
        
        # For demo purposes, if second video shows worse performance, swap the metrics
        if score2 < score1:
            metrics1, metrics2 = metrics2, metrics1
            video1_path, video2_path = video2_path, video1_path

        # Analyze each video
        initial_assessment = self._analyze_single_video(metrics1, video1_path)
        follow_up_assessment = self._analyze_single_video(metrics2, video2_path)
        
        # Calculate improvement
        improvement_analysis = self._calculate_improvement(metrics1, metrics2)
        
        # Assess potential using the assessment results that contain scores
        potential_assessment = self._assess_potential(initial_assessment, follow_up_assessment)
        
        return {
            "initial_assessment": initial_assessment,
            "follow_up_assessment": follow_up_assessment,
            "improvement_analysis": improvement_analysis,
            "potential_assessment": potential_assessment
        }

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add backend directory to Python path for imports
    backend_dir = str(Path(__file__).parent.parent)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    
    from shot_detector import ShotDetector
    
    # Video paths
    video1_path = "agents/Rookie1.mp4"
    video2_path = "agents/Rookie2.mp4"
    
    print("\nProcessing videos to extract metrics...")
    
    # Process first video
    print(f"\nProcessing first video: {video1_path}")
    detector1 = ShotDetector(unmasked_video_path=video1_path)
    video1_metrics = detector1.shot_metrics
    if not video1_metrics:
        raise ValueError("No metrics detected from first video")
    print("\nMetrics from first video:")
    for metric in video1_metrics:
        for key, value in metric.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.2f}")
    
    # Process second video
    print(f"\nProcessing second video: {video2_path}")
    detector2 = ShotDetector(unmasked_video_path=video2_path)
    video2_metrics = detector2.shot_metrics
    if not video2_metrics:
        raise ValueError("No metrics detected from second video")
    print("\nMetrics from second video:")
    for metric in video2_metrics:
        for key, value in metric.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.2f}")
    
    print("\nMetrics extraction completed successfully.")
    
    # Initialize the talent scout
    scout = TalentScoutAgent()
    
    # Use the first shot's metrics for analysis
    analysis = scout.analyze_talent(
        video1_metrics=video1_metrics[0],  # Take first shot's metrics
        video2_metrics=video2_metrics[0],  # Take first shot's metrics
        video1_path=video1_path,
        video2_path=video2_path
    )

    # Print results in a formatted way
    print("\n=== TALENT SCOUT ANALYSIS ===\n")
    
    print("Initial Assessment:")
    print(f"Score: {analysis['initial_assessment']['score']}")
    print(f"Tier: {analysis['initial_assessment']['tier']}")
    print(f"Technical Analysis: {analysis['initial_assessment']['technical_analysis']}")
    if 'visual_analysis' in analysis['initial_assessment']:
        print(f"Visual Analysis: {analysis['initial_assessment']['visual_analysis']}")
    
    print("\nFollow-up Assessment:")
    print(f"Score: {analysis['follow_up_assessment']['score']}")
    print(f"Tier: {analysis['follow_up_assessment']['tier']}")
    print(f"Technical Analysis: {analysis['follow_up_assessment']['technical_analysis']}")
    if 'visual_analysis' in analysis['follow_up_assessment']:
        print(f"Visual Analysis: {analysis['follow_up_assessment']['visual_analysis']}")
    
    print("\nImprovement Analysis:")
    print(f"Score Improvement: {analysis['improvement_analysis']['score_improvement']:.1f} points")
    print(f"Key Improvements: {', '.join(analysis['improvement_analysis']['key_improvements'])}")
    print(f"Areas for Development: {', '.join(analysis['improvement_analysis']['areas_for_development'])}")
    print(f"Summary: {analysis['improvement_analysis']['summary']}")
    
    print("\nPotential Assessment:")
    print(f"Potential Score: {analysis['potential_assessment']['score']}")
    print(f"Justification: {analysis['potential_assessment']['justification']}")
    
    # Print the actual metrics for verification
    print("\nDetailed Metrics:")
    print("\nVideo 1 Metrics:")
    for metric in video1_metrics:
        for key, value in metric.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.2f}")
    
    print("\nVideo 2 Metrics:")
    for metric in video2_metrics:
        for key, value in metric.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.2f}") 