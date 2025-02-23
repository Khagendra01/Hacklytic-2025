import React from 'react';
import { VideoAnalysis } from '../types';
import { AlertCircle } from 'lucide-react';

interface Props {
  analysis: VideoAnalysis;
}

export function VideoPlayer({ analysis }: Props) {
  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Original Video */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xl font-semibold text-gray-900">Original Video</h3>
            <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm">
              Original
            </span>
          </div>
          <div className="rounded-xl overflow-hidden bg-black aspect-video shadow-lg">
            <video
              src={analysis.maskedUrl}
              controls
              className="w-full h-full"
            />
          </div>
          <p className="text-sm text-gray-600">Original footage before pose estimation</p>
        </div>

        {/* Processed Video */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xl font-semibold text-gray-900">Pose Analysis</h3>
            <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm">
              Analysis Complete
            </span>
          </div>
          <div className="rounded-xl overflow-hidden bg-black aspect-video shadow-lg">
            <video
              src={analysis.processedUrl}
              controls
              className="w-full h-full"
            />
          </div>
          <p className="text-sm text-gray-600">Video with detected poses and movement analysis</p>
        </div>
      </div>
      
      {analysis.status === 'processing' && (
        <div className="flex items-center gap-3 text-yellow-700 bg-yellow-50 p-6 rounded-xl">
          <AlertCircle className="w-6 h-6" />
          <p className="text-lg">Pose estimation in progress. Analysis will be available soon.</p>
        </div>
      )}
    </div>
  );
}