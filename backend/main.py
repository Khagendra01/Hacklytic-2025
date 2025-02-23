from typing import Union
from agents.quick_analysis_agent import get_quick_analysis
from shot_detector import ShotDetector
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import aiohttp
import os
import uuid
from preprocessing.noise_masking import NoiseReducer, MaskingConfig
from firebase.firebase_storage import FirebaseStorageManager
import asyncio
from pydantic import BaseModel

app = FastAPI()

# Initialize Firebase Storage Manager
firebase_manager = FirebaseStorageManager(
    bucket_name='hacklytic2025.firebasestorage.app'
)

# Initialize NoiseReducer
noise_reducer = NoiseReducer(MaskingConfig())

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# Add this class before the endpoint
class VideoRequest(BaseModel):
    video_url: str

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/api/mask_video")
async def mask_video(request: VideoRequest):
    try:
        # Create temporary directory if it doesn't exist
        temp_dir = "temp_videos"
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Created/verified temp directory at: {os.path.abspath(temp_dir)}")
        
        video_url = request.video_url
        
        # Generate unique filenames
        input_video_path = os.path.join(temp_dir, f"input_{uuid.uuid4()}.mp4")
        print(f"Will save video to: {os.path.abspath(input_video_path)}")
        
        # Download the video
        async with aiohttp.ClientSession() as session:
            async with session.get(video_url) as response:
                if response.status != 200:
                    raise HTTPException(status_code=400, detail="Failed to download video")
                
                content = await response.read()
                print(f"Downloaded video content size: {len(content)} bytes")
                
                with open(input_video_path, 'wb') as f:
                    f.write(content)
                
                print(f"Video saved. File exists: {os.path.exists(input_video_path)}")
                print(f"File size: {os.path.getsize(input_video_path)} bytes")

        print(f"Input video path: {input_video_path}")
        
        # Process the video

        video_id = uuid.uuid4()
        unmasked_video_path = os.path.join(temp_dir, f"masked_{video_id}.mp4")
        await noise_reducer.process_video(input_video_path, unmasked_video_path)

        print(f"Output video path: {unmasked_video_path}")
        
        # Upload masked video to Firebase
        try:
            firebase_unmasked_path = f"masked_videos/{os.path.basename(unmasked_video_path)}"
            firebase_unmasked_url = firebase_manager.upload_file(unmasked_video_path, firebase_unmasked_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload masked video: {str(e)}")
        
        print(f"Masked video uploaded to Firebase: {firebase_unmasked_url}")

        # Process with shot detector
        try:
            output_file_dir = os.path.join(temp_dir, f"processed_{video_id}.mp4")
            detector = ShotDetector(unmasked_video_path=unmasked_video_path, output_file_dir=output_file_dir)
            shot_metrics = detector.shot_metrics

            # Upload processed video
            firebase_processed_path = f"processed_videos/{os.path.basename(output_file_dir)}"
            firebase_processed_url = firebase_manager.upload_file(output_file_dir, firebase_processed_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process or upload analyzed video: {str(e)}")
        
        analysis = get_quick_analysis(metrics_data=shot_metrics, video_url=output_file_dir)
        
        return {
            "masked_video_url": firebase_unmasked_url,
            "processed_video_url": firebase_processed_url,
            "shot_metrics": shot_metrics,
            "analysis": analysis
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up temporary files
        for path in [input_video_path, unmasked_video_path, output_file_dir]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Failed to remove temporary file {path}: {e}")

@app.get("/test_firebase")
async def test_firebase():
    try:
        test_file = "output_video.mp4"
        if not os.path.exists(test_file):
            raise HTTPException(status_code=404, detail="Test file not found")
            
        # Try uploading to Firebase
        firebase_path = f"test_uploads/{os.path.basename(test_file)}"
        public_url = firebase_manager.upload_file(test_file, firebase_path)
        
        return {
            "status": "success",
            "message": "Test file uploaded successfully",
            "url": public_url
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Firebase test failed: {str(e)}")
