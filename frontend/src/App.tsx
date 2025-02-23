import React, { useState } from 'react';
import { Activity } from 'lucide-react';
import { VideoUploader } from './components/VideoUploader';
import { VideoPlayer } from './components/VideoPlayer';
import { uploadVideo } from './api';
import { VideoAnalysis } from './types';

function App() {
  const [isUploading, setIsUploading] = useState(false);
  const [analysis, setAnalysis] = useState<VideoAnalysis | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleUpload = async (file: File) => {
    try {
      setError(null);
      setIsUploading(true);
      const result = await uploadVideo(file);
      setAnalysis(result);
      
      // Poll for updates
      const interval = setInterval(() => {
        if (result.status === 'completed') {
          setAnalysis({ ...result });
          clearInterval(interval);
        }
      }, 1000);
      
    } catch (err) {
      setError('Failed to upload video. Please try again.');
      console.error('Upload failed:', err);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Activity className="w-8 h-8 text-blue-600" />
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-blue-800 text-transparent bg-clip-text">
                CourtVision AI
              </h1>
            </div>
            {error && (
              <div className="text-red-500 text-sm flex items-center gap-2">
                <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></span>
                {error}
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="max-w-4xl mx-auto space-y-8">
          <div className="text-center space-y-4">
            <h2 className="text-4xl font-bold text-gray-900 sm:text-5xl">
              Basketball Video Analysis
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Upload your basketball footage and let our AI analyze player movements,
              shot accuracy, and game patterns in real-time.
            </p>
          </div>

          <div className="bg-white rounded-2xl shadow-xl p-8 border">
            {!analysis ? (
              <VideoUploader onUpload={handleUpload} isUploading={isUploading} />
            ) : (
              <VideoPlayer analysis={analysis} />
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white/80 backdrop-blur-sm border-t mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-gray-500">
            © 2025 CourtVision AI. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;