import React from 'react';
import { VideoAnalysis } from '../types';
import { AlertCircle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface Props {
  analysis: VideoAnalysis;
}

export function VideoPlayer({ analysis }: Props) {
  console.log(analysis);

  const formatMarkdownText = (text: string) => {
    // First handle the bold text
    let formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Handle bullet points - look for lines starting with *
    formattedText = formattedText.split('\n').map(line => {
      const trimmedLine = line.trim();
      if (trimmedLine.startsWith('* ')) {
        // Extract the content after the bullet point
        const content = trimmedLine.substring(2).trim();
        return `<div class="flex items-start gap-2 mb-2">
          <span class="mt-1.5 h-1.5 w-1.5 rounded-full bg-gray-500 flex-shrink-0"></span>
          <span>${content}</span>
        </div>`;
      }
      return line;
    }).join('\n');

    return formattedText;
  };

  const renderMarkdown = (text: string) => {
    return text.split('|').map((section, index) => {
      const trimmedSection = section.trim();
      if (!trimmedSection) return null;

      // Format the markdown text
      const formattedText = formatMarkdownText(trimmedSection);

      return (
        <div key={index} className="mb-6 last:mb-0">
          <div 
            className="markdown-content space-y-2"
            dangerouslySetInnerHTML={{ __html: formattedText }}
          />
        </div>
      );
    }).filter(Boolean);
  };

  return (
    <div className="space-y-8">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-xl font-semibold text-gray-900">Original Video</h3>
          <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm">
            Original
          </span>
        </div>
        <div className="rounded-xl overflow-hidden bg-black aspect-video shadow-lg max-w-3xl mx-auto">
          <video
            src={analysis.masked_video_url}
            controls
            className="w-full h-full"
          />
        </div>
        <p className="text-sm text-gray-600 text-center">Original footage before pose estimation</p>
      </div>

      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-xl font-semibold text-gray-900">Pose Analysis</h3>
          <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm">
            Analysis Complete
          </span>
        </div>
        <div className="rounded-xl overflow-hidden bg-black aspect-video shadow-lg max-w-3xl mx-auto">
          <video
            src={analysis.processed_video_url}
            controls
            className="w-full h-full"
          />
        </div>
        <p className="text-sm text-gray-600 text-center">Video with detected poses and movement analysis</p>
      </div>

      {/* Analysis Summary Box */}
      {analysis.analysis && (
        <div className="max-w-3xl mx-auto bg-white rounded-xl shadow-lg p-6 border border-gray-200">
          <h3 className="text-xl font-semibold text-gray-900 mb-4">Analysis Summary</h3>
          {analysis.analysis.coach_recommendations && (
            <div className="prose prose-blue max-w-none mb-6 text-gray-600">
              {renderMarkdown(analysis.analysis.coach_recommendations)}
            </div>
          )}
          {analysis.analysis.coach_reasoning && (
            <div className="prose prose-blue max-w-none text-gray-600 [&_strong]:font-semibold [&_strong]:text-gray-900">
              {renderMarkdown(analysis.analysis.coach_reasoning)}
            </div>
          )}
        </div>
      )}

      {analysis.status === 'processing' && (
        <div className="flex items-center gap-3 text-yellow-700 bg-yellow-50 p-6 rounded-xl">
          <AlertCircle className="w-6 h-6" />
          <p className="text-lg">Pose estimation in progress. Analysis will be available soon.</p>
        </div>
      )}
    </div>
  );
}