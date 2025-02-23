import os
import uuid
import aiohttp
from fastapi import HTTPException
from backend.shot_detector import ShotDetector
from backend.noise_reducer import NoiseReducer
from backend.firebase_manager import firebase_manager
async def process_video(video_url: str):
    try:
        # Create temporary directory if it doesn't exist
        temp_dir = "temp_videos"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate unique filenames for input and output videos
        input_video_path = os.path.join(temp_dir, f"input_{uuid.uuid4()}.mp4")
        output_video_path = os.path.join(temp_dir, f"output_{uuid.uuid4()}.mp4")
        
        # Download the video
        async with aiohttp.ClientSession() as session:
            async with session.get(video_url) as response:
                if response.status != 200:
                    raise HTTPException(status_code=400, detail="Failed to download video")
                
                with open(input_video_path, 'wb') as f:
                    f.write(await response.read())
        
        # Process the video
        await noise_reducer.process_video(input_video_path, output_video_path)
        
        # Upload to Firebase
        firebase_path = f"masked_videos/{os.path.basename(output_video_path)}"
        public_url = firebase_manager.upload_file(output_video_path, firebase_path)

        detector = ShotDetector(output_video_path)
        processed_video_path = f'processed_videos/{os.path.basename(output_video_path)}'
        shot_metrics = detector.shot_metrics

        firebase_processed_path = f"processed_videos/{os.path.basename(output_video_path)}"
        public_processed_url = firebase_manager.upload_file(processed_video_path, firebase_processed_path)


        
        # Clean up temporary files
        os.remove(input_video_path)
        os.remove(output_video_path)
        os.remove(processed_video_path)

        # # Upload shot metrics to Firebase
        # firebase_shot_metrics_path = f"shot_metrics/{os.path.basename(output_video_path)}"
        # public_shot_metrics_url = firebase_manager.upload_file(shot_metrics, firebase_shot_metrics_path)

        
        return {"masked_video_url": public_url, "processed_video_url": public_processed_url, "shot_metrics": shot_metrics}
        
    except Exception as e:
        # Clean up temporary files in case of error
        if 'input_video_path' in locals() and os.path.exists(input_video_path):
            os.remove(input_video_path)
        if 'output_video_path' in locals() and os.path.exists(output_video_path):
            os.remove(output_video_path)
        raise HTTPException(status_code=500, detail=str(e))