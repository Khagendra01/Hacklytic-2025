from typing import Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import aiohttp
import os
import uuid
from preprocessing.noise_masking import NoiseReducer, MaskingConfig
from firebase.firebase_storage import FirebaseStorageManager
import asyncio

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

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/api/mask_video")
async def mask_video(video_url: str):
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
        
        # Clean up temporary files
        os.remove(input_video_path)
        os.remove(output_video_path)
        
        return {"masked_video_url": public_url}
        
    except Exception as e:
        # Clean up temporary files in case of error
        if 'input_video_path' in locals() and os.path.exists(input_video_path):
            os.remove(input_video_path)
        if 'output_video_path' in locals() and os.path.exists(output_video_path):
            os.remove(output_video_path)
        raise HTTPException(status_code=500, detail=str(e))

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
