import React, { useState, ChangeEvent } from 'react';
import { storage } from '../firebaseConfig';
import { ref, uploadBytes, getDownloadURL } from 'firebase/storage';
import { analyzeVideos } from '../api';

interface VideoUpload {
  file: File | null;
  timeframe: string;
}

interface Assessment {
  score: number;  // Score between 40-100
  tier: 'S' | 'A' | 'B' | 'C';  // Defined tiers
  technical_analysis: string;  // Detailed analysis of form and metrics
  visual_analysis?: string;  // Optional: Only present if video_path is provided
}

interface ImprovementAnalysis {
  score_improvement: number;  // Points improved
  key_improvements: string[];  // List of metrics that improved
  areas_for_development: string[];  // List of metrics needing work
  summary: string;  // Text summary of improvements
}

interface PotentialAssessment {
  score: number;  // Potential ceiling score
  justification: string;  // Explanation of potential calculation
}

interface AnalysisResult {
  initial_assessment: Assessment;
  follow_up_assessment: Assessment;
  improvement_analysis: ImprovementAnalysis;
  potential_assessment: PotentialAssessment;
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
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>()
  
  const timeOptions = ['6 months', '1 year', '2 years', '3 years', '4 years', '5 years'];

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
        
        {/* Upload Section */}
        <div className="space-y-6">
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-600 px-4 py-3 rounded-lg text-center">
              {error}
            </div>
          )}

          <div className="grid md:grid-cols-2 gap-6">
            {uploads.map((upload, index) => (
              <div key={index} className="space-y-4">
                <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
                  <h3 className="font-semibold text-lg mb-4">
                    {index === 0 ? 'Initial Video' : 'Follow-up Video'}
                  </h3>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Upload Video
                      </label>
                      <div className="relative">
                        <input
                          type="file"
                          className="hidden"
                          accept="video/*"
                          onChange={handleFileChange(index)}
                          id={`video-upload-${index}`}
                        />
                        <label
                          htmlFor={`video-upload-${index}`}
                          className={`
                            flex items-center justify-center w-full px-4 py-3 rounded-lg
                            border-2 border-dashed transition-colors cursor-pointer
                            ${upload.file 
                              ? 'border-green-300 bg-green-50 hover:bg-green-100' 
                              : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50'
                            }
                          `}
                        >
                          <div className="text-center">
                            {upload.file ? (
                              <div className="flex items-center space-x-2">
                                <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                                </svg>
                                <span className="text-sm text-gray-600">{upload.file.name}</span>
                              </div>
                            ) : (
                              <>
                                <svg className="mx-auto h-8 w-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                                </svg>
                                <span className="mt-2 block text-sm text-gray-600">
                                  Click to upload or drag and drop
                                </span>
                              </>
                            )}
                          </div>
                        </label>
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Time Period
                      </label>
                      <select 
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
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
                  </div>
                </div>
              </div>
            ))}
          </div>
                
          <button
            onClick={handleCompute}
            disabled={isComputing}
            className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors disabled:bg-blue-400 disabled:cursor-not-allowed font-medium text-sm flex items-center justify-center space-x-2"
          >
            {isComputing ? (
              <>
                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Processing...</span>
              </>
            ) : (
              <>
                <span>Analyze Videos</span>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
                </svg>
              </>
            )}
          </button>
        </div>

        {/* Analysis Result Card */}
        {analysisResult && (
          <div className="mt-8 bg-white rounded-xl shadow-lg overflow-hidden">
            <div className="p-6">
              <div className="space-y-6">
                {/* Initial Assessment */}
                <div>
                  <h3 className="text-2xl font-bold mb-4">Initial Assessment</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="font-semibold">Score: {analysisResult.initial_assessment.score}</p>
                      <p className="font-semibold">Tier: {analysisResult.initial_assessment.tier}</p>
                    </div>
                    <div>
                      <p className="text-gray-700">{analysisResult.initial_assessment.technical_analysis}</p>
                      {analysisResult.initial_assessment.visual_analysis && (
                        <p className="text-gray-700 mt-2">{analysisResult.initial_assessment.visual_analysis}</p>
                      )}
                    </div>
                  </div>
                </div>

                {/* Follow-up Assessment */}
                <div>
                  <h3 className="text-2xl font-bold mb-4">Follow-up Assessment</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="font-semibold">Score: {analysisResult.follow_up_assessment.score}</p>
                      <p className="font-semibold">Tier: {analysisResult.follow_up_assessment.tier}</p>
                    </div>
                    <div>
                      <p className="text-gray-700">{analysisResult.follow_up_assessment.technical_analysis}</p>
                      {analysisResult.follow_up_assessment.visual_analysis && (
                        <p className="text-gray-700 mt-2">{analysisResult.follow_up_assessment.visual_analysis}</p>
                      )}
                    </div>
                  </div>
                </div>

                {/* Improvement Analysis */}
                <div>
                  <h3 className="text-2xl font-bold mb-4">Improvement Analysis</h3>
                  <p className="font-semibold text-green-600">Score Improvement: {analysisResult.improvement_analysis.score_improvement} points</p>
                  <div className="mt-4">
                    <h4 className="font-semibold">Key Improvements:</h4>
                    <ul className="list-disc pl-5">
                      {analysisResult.improvement_analysis.key_improvements.map((improvement, index) => (
                        <li key={index}>{improvement}</li>
                      ))}
                    </ul>
                  </div>
                  <div className="mt-4">
                    <h4 className="font-semibold">Areas for Development:</h4>
                    <ul className="list-disc pl-5">
                      {analysisResult.improvement_analysis.areas_for_development.map((area, index) => (
                        <li key={index}>{area}</li>
                      ))}
                    </ul>
                  </div>
                  <p className="mt-4 text-gray-700">{analysisResult.improvement_analysis.summary}</p>
                </div>

                {/* Potential Assessment */}
                <div>
                  <h3 className="text-2xl font-bold mb-4">Potential Assessment</h3>
                  <p className="font-semibold">Potential Score: {analysisResult.potential_assessment.score}</p>
                  <p className="text-gray-700 mt-2">{analysisResult.potential_assessment.justification}</p>
                </div>
              </div>
            </div>
          </div>
        )}

      </div>
    </main>
  );
} 