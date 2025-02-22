import os
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool

@dataclass
class ShootingMetrics:
    elbow_angle: float
    wrist_angle: float
    shoulder_angle: float
    knee_angle: float
    shot_trajectory: float
    release_height_ratio: float
    release_height: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShootingMetrics':
        return cls(
            elbow_angle=float(data.get('Elbow Angle', '0').split('°')[0]),
            wrist_angle=float(data.get('Wrist Angle', '0').split('°')[0]),
            shoulder_angle=float(data.get('Shoulder Angle', '0').split('°')[0]),
            knee_angle=float(data.get('Knee Angle', '0').split('°')[0]),
            shot_trajectory=float(data.get('Shot Trajectory', '0').split('°')[0]),
            release_height_ratio=float(data.get('Release Height Ratio', '0')),
            release_height=float(data.get('RELEASE HEIGHT', '0'))
        )

class ShootingScoreAnalyzer:
    def __init__(self):
        self.ideal_ranges = {
            'elbow_angle': (165, 175),
            'wrist_angle': (70, 90),
            'shoulder_angle': (90, 110),
            'knee_angle': (140, 170),
            'shot_trajectory': (45, 55),
            'release_height_ratio': (1.8, 2.2)
        }

    def calculate_score(self, metrics: ShootingMetrics) -> Dict[str, Any]:
        scores = {}
        reasoning = []

        # Score elbow angle
        scores['elbow_angle_score'] = self._score_metric(
            metrics.elbow_angle,
            self.ideal_ranges['elbow_angle'],
            "Elbow angle",
            reasoning
        )

        # Score wrist angle
        scores['wrist_angle_score'] = self._score_metric(
            metrics.wrist_angle,
            self.ideal_ranges['wrist_angle'],
            "Wrist angle",
            reasoning
        )

        # Score shoulder angle
        scores['shoulder_angle_score'] = self._score_metric(
            metrics.shoulder_angle,
            self.ideal_ranges['shoulder_angle'],
            "Shoulder angle",
            reasoning
        )

        # Score knee angle
        scores['knee_angle_score'] = self._score_metric(
            metrics.knee_angle,
            self.ideal_ranges['knee_angle'],
            "Knee angle",
            reasoning
        )

        # Score shot trajectory
        scores['shot_trajectory_score'] = self._score_metric(
            metrics.shot_trajectory,
            self.ideal_ranges['shot_trajectory'],
            "Shot trajectory",
            reasoning
        )

        # Score release height ratio
        scores['release_height_ratio_score'] = self._score_metric(
            metrics.release_height_ratio,
            self.ideal_ranges['release_height_ratio'],
            "Release height ratio",
            reasoning
        )

        # Calculate overall score
        scores['overall_score'] = int(sum(scores.values()) / len(scores))
        
        # Add reasoning
        scores['reasoning'] = " ".join(reasoning)

        return scores

    def _score_metric(self, value: float, ideal_range: tuple, metric_name: str, reasoning: List[str]) -> int:
        min_val, max_val = ideal_range
        
        # Perfect score if within ideal range
        if min_val <= value <= max_val:
            reasoning.append(f"{metric_name} is optimal ({value:.1f}).")
            return 10
            
        # Calculate how far outside the range
        if value < min_val:
            deviation = min_val - value
            reference = min_val
        else:
            deviation = value - max_val
            reference = max_val
            
        # Score decreases based on how far from ideal range
        percentage_off = deviation / reference
        score = max(0, int(10 - (percentage_off * 20)))
        
        reasoning.append(
            f"{metric_name} ({value:.1f}) is {'below' if value < min_val else 'above'} "
            f"ideal range ({min_val}-{max_val}). Score: {score}/10."
        )
        
        return score

def initialize_agent():
    """Initialize the shooting form analysis agent."""
    llm = ChatOpenAI(model="gpt-4")
    analyzer = ShootingScoreAnalyzer()
    
    def analyze_shooting_form(input_data: Dict[str, Any]) -> Dict[str, Any]:
        results = {
            "player_scores": [],
            "user_scores": None
        }
        
        # Analyze other players' data
        if "Players_data" in input_data:
            for player_data in input_data["Players_data"]:
                metrics = ShootingMetrics.from_dict(player_data)
                scores = analyzer.calculate_score(metrics)
                results["player_scores"].append(scores)
        
        # Analyze user data
        if "User_data" in input_data:
            metrics = ShootingMetrics.from_dict(input_data["User_data"])
            results["user_scores"] = analyzer.calculate_score(metrics)
        
        return results

    tools = [
        Tool(
            name="analyze_shooting_form",
            func=analyze_shooting_form,
            description="Analyzes basketball shooting form metrics and provides detailed scoring"
        )
    ]

    memory = MemorySaver()
    config = {"configurable": {"thread_id": "Shooting Form Analysis Agent"}}

    return create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        state_modifier=(
            "You are a basketball shooting form analysis agent specialized in evaluating shooting mechanics. "
            "Your task is to analyze shooting form metrics and provide detailed scoring and feedback.\n\n"
            "Always format your response as a JSON object with scores and reasoning for each metric."
        ),
    ), config

def main():
    # Test data
    test_data = {
        "Players_data": [{
            "Elbow Angle": "131.9°",
            "Wrist Angle": "273.9°",
            "Shoulder Angle": "143.5°",
            "Knee Angle": "168.6°",
            "Shot Trajectory": "-19.0°",
            "Release Height Ratio": "2.05",
            "RELEASE HEIGHT": "-210.5"
        }],
        "User_data": {
            "Elbow Angle": "170.5°",
            "Wrist Angle": "85.2°",
            "Shoulder Angle": "95.5°",
            "Knee Angle": "155.6°",
            "Shot Trajectory": "48.0°",
            "Release Height Ratio": "1.95",
            "RELEASE HEIGHT": "-180.5"
        }
    }

    agent_executor, config = initialize_agent()

    prompt = (
        f"Please analyze the shooting form metrics for these players:\n\n"
        f"{json.dumps(test_data, indent=2)}\n\n"
        "Provide detailed scoring and feedback for each metric."
    )

    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content=prompt)]},
        config
    ):
        if "agent" in chunk:
            print(chunk["agent"]["messages"][0].content)
        elif "tools" in chunk:
            print(chunk["tools"]["messages"][0].content)

if __name__ == "__main__":
    main() 