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

    // const response = {
    //   maskedUrl: 'https://firebasestorage.googleapis.com/v0/b/hacklytic2025.firebasestorage.app/o/videos%2Fmore3.mp4?alt=media&token=9dede87d-90c6-406d-ba2c-80d8bebd1662',
    //   status: 'completed',
    //   processedUrl: 'https://firebasestorage.googleapis.com/v0/b/hacklytic2025.firebasestorage.app/o/videos%2Fmore7.mp4?alt=media&token=14842411-2332-415d-9c14-817d9e216ae9'
    // }

    const analysis: VideoAnalysis = response;
    return analysis;
  } catch (error) {
    console.error('Upload failed:', error);
    throw error;
  }
}