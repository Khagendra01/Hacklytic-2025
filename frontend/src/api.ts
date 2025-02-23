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
      masked_video_url: data.masked_video_url,
      processed_video_url: data.processed_video_url,
      status: 'completed',
      shot_metrics: data.shot_metrics || [],
      analysis: data.analysis
    };
  } catch (error) {
    console.error('Upload failed:', error);
    throw error;
  }
}

interface VideoAnalysisRequest {
  url: string;
  timeframe: string;
}

export const analyzeVideos = async (videos: VideoAnalysisRequest[]) => {
  const response = await fetch(`${API_URL}/talent_scout`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ video1_url: videos[0].url, video2_url: videos[1].url }),
  });

  if (!response.ok) {
    throw new Error('Failed to analyze videos');
  }

  return response.json();
};