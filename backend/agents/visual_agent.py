import os
import sys
from PIL import Image
import requests
from io import BytesIO
from typing import List
import json
from transformers import CLIPProcessor, CLIPModel
import torch
import pandas as pd
from datetime import datetime
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent  # This is the correct import
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
from dotenv import load_dotenv
import asyncio
# Setup environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
env_path = os.path.join(os.path.dirname(__file__), '../../../sepolia/.env.local')
load_dotenv(env_path)

class VisualImpactAnalyzer:
    def __init__(self):
        load_dotenv()
        self.pinata_jwt = os.getenv('PINATA_JWT')
        self.gateway_url = os.getenv('NEXT_PUBLIC_GATEWAY_URL')
        if not self.pinata_jwt or not self.gateway_url:
            raise ValueError("PINATA_JWT or NEXT_PUBLIC_GATEWAY_URL not found in environment variables")

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Aesthetic attributes to evaluate
        self.attributes = [
            "high quality", "professional", "stunning", "eye-catching",
            "dramatic", "vibrant", "well-composed", "visually striking",
            "artistic", "creative", "unique", "memorable"
        ]

    def convert_cid_to_url(self, cid: str) -> str:
        """Convert IPFS CID to a Pinata gateway URL."""
        return f"https://{self.gateway_url}/ipfs/{cid}"

    def download_and_convert_to_pil(self, cid: str) -> Image.Image:
        """Download image from IPFS CID using Pinata gateway and convert to PIL Image."""
        try:
            image_url = self.convert_cid_to_url(cid)
            print(f"Attempting to download image from: {image_url}")
            
            headers = {
                'Authorization': f'Bearer {self.pinata_jwt}',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(image_url, headers=headers, timeout=10)
            print(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error response content: {response.content}")
                return None
                
            image = Image.open(BytesIO(response.content))
            print(f"Successfully opened image: {image.size}")
            return image
        except Exception as e:
            print(f"Error downloading image: {str(e)}")
            print(f"Full error details: {type(e).__name__}")
            return None

    def analyze_single_image(self, image: Image.Image) -> float:
        """Analyze a single image and return impact score."""
        inputs = self.processor(
            images=image,
            text=self.attributes,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds

            similarity = torch.nn.functional.cosine_similarity(
                image_features[:, None, :],
                text_features[None, :, :],
                dim=-1
            )

            # Convert similarity scores to 0-100 scale
            average_score = float(similarity.mean() * 100)
            return min(max(average_score, 0), 100)

    def analyze_impact(self, cids: List[str]) -> List[int]:
        """Analyze multiple images from IPFS CIDs and return list of impact scores."""
        # Ensure cids is treated as a list, not a string
        if isinstance(cids, str):
            cids = [cids]  # If a single CID is passed as string, convert to list
        
        print(f"Received CIDs: {cids}")
        print(f"Number of CIDs: {len(cids)}")
        
        scores = []
        for cid in cids:
            print(f"\nProcessing CID: {cid}")
            image = self.download_and_convert_to_pil(cid)
            if image:
                score = self.analyze_single_image(image)
                scores.append(int(round(score)))
                print(f"Successfully analyzed image. Score: {scores[-1]}")
            else:
                scores.append(0)  # Default score for failed downloads
                print("Failed to process image, assigned default score: 0")
        return scores

def initialize_agent():
    """Initialize the visual impact analysis agent."""
    llm = ChatOpenAI(model="gpt-4")
    
    visual_analyzer = VisualImpactAnalyzer()

    tools = [
        Tool(
            name="analyze_visual_impact",
            func=visual_analyzer.analyze_impact,
            description="Analyzes visual impact of NFT images and returns impact scores"
        )
    ]

    memory = MemorySaver()
    config = {"configurable": {"thread_id": "Visual Impact Analysis Agent"}}

    return create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        state_modifier=(
            "You are a visual analysis agent specialized in evaluating NFT images. "
            "Your task is to analyze the visual impact and artistic quality of NFT images. "
            "For each image, provide a detailed analysis and impact score.\n\n"
            "Always format your response as a JSON object with this structure:\n"
            "{\n"
            '    "image_analyses": [\n'
            '        {\n'
            '            "cid": "IPFS CID of the image",\n'
            '            "impact_score": "0-100 numerical score",\n'
            '            "analysis": "Detailed visual analysis of the image"\n'
            '        }\n'
            '    ],\n'
            '    "overall_assessment": "Summary of all images analyzed"\n'
            "}"
        ),
    ), config

def main():
    # Test CIDs - replace with actual NFT CIDs you want to analyze
    mock_nft_cids = [
        "bafkreie44ehpnzcfupb46r5jsd5gs236oozpeidcp2qqou3hwxw7fj5pui"  # Replace with real CIDs
    ]

    agent_executor, config = initialize_agent()

    prompt = (
        f"Please analyze the visual impact of these NFT images (IPFS CIDs):\n\n"
        f"{json.dumps(mock_nft_cids, indent=2)}\n\n"
        "Provide impact scores and detailed analysis for each image."
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