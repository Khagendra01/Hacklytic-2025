import { VideoAnalysis } from './types';
import { storage } from './firebaseConfig';
import { ref, uploadBytes, getDownloadURL } from 'firebase/storage';

const API_URL = 'http://localhost:8000/api';

export async function uploadVideo(file: File): Promise<VideoAnalysis> {
  try {
    const filename = `${Date.now()}-${file.name}`;
    const videoRef = ref(storage, `videos/${filename}`);
    const uploadResult = await uploadBytes(videoRef, file);
    const downloadURL = await getDownloadURL(videoRef);
    const response = await fetch(`${API_URL}/mask_video`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        video_url: downloadURL,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to process video');
    }

    const data = await response.json();
    
    return {
      maskedUrl: data.masked_video_url,
      processedUrl: data.processed_video_url,
      status: 'completed',
      shotMetrics: data.shot_metrics || []
    };
  } catch (error) {
    console.error('Upload failed:', error);
    throw error;
  }
}