import React, { useState, ChangeEvent } from 'react';
import { storage } from '../firebaseConfig';
import { ref, uploadBytes, getDownloadURL } from 'firebase/storage';
import { analyzeVideos } from '../api';

interface VideoUpload {
  file: File | null;
  timeframe: string;
}

interface AnalysisResult {
  img_url: string;
  rank: string;

}

interface VideoData {
  url: string;
  timeframe: string;
}

interface DetailedAnalysis {
  larry: {
    cards: {
      c1: { img_url: string; rank: string, url: string };
      c2: { img_url: string; rank: string, url: string };
      c3: { img_url: string; rank: string, url: string };
    };
  };
  rookie: {
    cards: {
      c1: { img_url: string; rank: string, url: string };
      c2: { img_url: string; rank: string, url: string };
    };
  };
}

export function ProPage() {
  const [uploads, setUploads] = useState<VideoUpload[]>([
    { file: null, timeframe: '6 months' },
    { file: null, timeframe: '6 months' }
  ]);
  const [isComputing, setIsComputing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>();
  const [showDetails, setShowDetails] = useState(false);
  const [detailedData] = useState<DetailedAnalysis>({
    larry: {
      cards: {
        c1: { img_url: "https://example.com/1", rank: "A", url: "video_url_1" },
        c2: { img_url: "https://example.com/2", rank: "B", url: "video_url_1" },
        c3: { img_url: "https://example.com/3", rank: "C", url: "video_url_1" }
      }
    },
    rookie: {
      cards: {
        c1: { img_url: "https://example.com/4", rank: "B", url: "video_url_1" },
        c2: { img_url: "https://example.com/5", rank: "C", url: "video_url_1" }
      }
    }
  });
  
  const timeOptions = ['6 months', '1 year'];

  const handleFileChange = (index: number) => (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setUploads(prev => prev.map((upload, i) => 
        i === index ? { ...upload, file: e.target.files![0] } : upload
      ));
    }
  };

  const handleTimeframeChange = (index: number) => (e: ChangeEvent<HTMLSelectElement>) => {
    setUploads(prev => prev.map((upload, i) => 
      i === index ? { ...upload, timeframe: e.target.value } : upload
    ));
  };

  const uploadToFirebase = async (file: File) => {
    const timestamp = Date.now();
    const storageRef = ref(storage, `videos/${timestamp}_${file.name}`);
    await uploadBytes(storageRef, file);
    return getDownloadURL(storageRef);
  };

  const handleCompute = async () => {
    setIsComputing(true);
    setError(null);
    setAnalysisResult(null); // Reset previous results

    // Validate files are selected
    if (!uploads[0].file || !uploads[1].file) {
      setError('Please select both video files');
      setIsComputing(false);
      return;
    }

    try {
      // Upload both files to Firebase
      const uploadPromises = uploads
        .filter(upload => upload.file)
        .map(async upload => {
          const url = await uploadToFirebase(upload.file!);
          return {
            url,
            timeframe: upload.timeframe
          };
        });

      const uploadedVideos = await Promise.all(uploadPromises);

      // Send URLs to backend
      const analysis = await analyzeVideos(uploadedVideos);
      setAnalysisResult(analysis);
      
    } catch (error) {
      console.error('Error:', error);
      setError('Failed to process videos. Please try again.');
    } finally {
      setIsComputing(false);
    }
  };

  return (
    <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="max-w-4xl mx-auto space-y-8">
        <h2 className="text-3xl font-bold text-center">Pro Analysis</h2>
        
        {/* Analysis Result Card */}
        {analysisResult && (
          <div className="mt-8 bg-white rounded-xl shadow-lg overflow-hidden">
            <div className="aspect-w-16 aspect-h-9">
              <img 
                src={analysisResult.img_url} 
                alt="Analysis Result"
                className="w-full h-full object-cover"
              />
            </div>
            <div className="p-6">
              <div className="flex items-center justify-between">
                <h3 className="text-2xl font-bold">Our Rookie</h3>
                <span className="text-xl font-semibold text-blue-600">
                  Rank: {analysisResult.rank}
                </span>
              </div>
              <button 
                className="mt-4 w-full bg-gray-800 text-white py-2 px-4 rounded-lg hover:bg-gray-900 transition-colors"
                onClick={() => setShowDetails(true)}
              >
                More Details
              </button>
            </div>
          </div>
        )}

        {showDetails && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white p-6 rounded-xl w-full max-w-6xl max-h-[90vh] overflow-y-auto">
              <div className="flex justify-between mb-6">
                <h2 className="text-2xl font-bold">Detailed Analysis</h2>
                <button 
                  onClick={() => setShowDetails(false)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  Close
                </button>
              </div>
              
              <div className="grid grid-cols-2 gap-8">
                {/* Larry's Section */}
                <div className="space-y-4">
                  <h3 className="text-xl font-semibold">Larry</h3>
                  <div className="space-y-4">

                    {Object.entries(detailedData.larry.cards).map(([key, card]) => (
                      <div key={key} className="border rounded-lg p-4">
                                            <div className="border rounded-lg p-4">
                      <video 
                        controls 
                        className="w-full rounded-lg"
                        src={card.url}
                      />
                    </div>
                    <div className="flex items-center gap-4 mt-2">
                        <img 
                          src={card.img_url} 
                          alt={`Analysis ${key}`}
                          className="w-24 h-24 object-cover rounded-lg"
                        />
                        <span className="font-semibold">Rank: {card.rank}</span>
                      </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Rookie's Section */}
                <div className="space-y-4">
                  <h3 className="text-xl font-semibold">Rookie</h3>
                  <div className="space-y-4">

                    {Object.entries(detailedData.rookie.cards).map(([key, card]) => (
                      <div key={key} className="border rounded-lg p-4">
                                            <div className="border rounded-lg p-4">
                      <video 
                        controls 
                        className="w-full rounded-lg"
                        src={card.url}
                      />
                    </div>
                    <div className="flex items-center gap-4 mt-2">
                          <img 
                            src={card.img_url} 
                            alt={`Analysis ${key}`}
                            className="w-24 h-24 object-cover rounded-lg"
                          />
                          <span className="font-semibold">Rank: {card.rank}</span>
                        </div>

                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="space-y-6">
          {error && (
            <div className="text-red-500 text-center">{error}</div>
          )}

          <div className="flex gap-4">
            {uploads.map((upload, index) => (
              <div key={index} className="flex-1">
                <input
                  type="file"
                  className="w-full p-2 border rounded-lg"
                  accept="video/*"
                  onChange={handleFileChange(index)}
                />
                <select 
                  className="mt-2 w-full p-2 border rounded-lg"
                  value={upload.timeframe}
                  onChange={handleTimeframeChange(index)}
                >
                  {timeOptions.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </div>
            ))}
          </div>
                  
          <button
            onClick={handleCompute}
            disabled={isComputing}
            className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 transition-colors disabled:bg-blue-400"
          >
            {isComputing ? 'Computing...' : 'Compute Analysis'}
          </button>
        </div>
      </div>
    </main>
  );
} 