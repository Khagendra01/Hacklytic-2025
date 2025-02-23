import React, { useCallback, useState } from 'react';
import { Upload, AlertCircle } from 'lucide-react';
import { clsx } from 'clsx';
import { ref, uploadBytesResumable } from "firebase/storage";
import { storage } from '../firebaseConfig'; // Adjust this import path as needed

interface Props {
  onUpload: (file: File) => void;
  isUploading: boolean;
}

export function VideoUploader({ onUpload, isUploading }: Props) {
  const [dragActive, setDragActive] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  
  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleFileUpload = useCallback((file: File) => {
    if (!file.type.startsWith('video/')) {
      setUploadError('Please upload a video file');
      return;
    }
    if (file.size > 1024 * 1024 * 1024) { // 1GB
      setUploadError('File size must be less than 1GB');
      return;
    }
    onUpload(file);
  }, [onUpload]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('video/')) {
      handleFileUpload(file);
    }
  }, [handleFileUpload]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  }, [handleFileUpload]);

  return (
    <div
      className={clsx(
        "relative rounded-xl border-3 border-dashed p-12 transition-all duration-200",
        dragActive ? "border-blue-500 bg-blue-50" : "border-gray-200 hover:border-blue-400",
        isUploading ? "opacity-50 cursor-not-allowed" : "cursor-pointer"
      )}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <input
        type="file"
        accept="video/*"
        onChange={handleChange}
        disabled={isUploading}
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
      />
      
      <div className="flex flex-col items-center justify-center text-center">
        <div className="w-20 h-20 rounded-full bg-blue-50 flex items-center justify-center mb-6">
          <Upload className="w-10 h-10 text-blue-500" />
        </div>
        <p className="text-2xl font-semibold text-gray-700">
          {isUploading ? "Processing your video..." : "Drop your video here"}
        </p>
        <p className="mt-3 text-lg text-gray-500">
          or click to select a file
        </p>
        <p className="mt-2 text-sm text-gray-400">
          Supports MP4, MOV up to 1GB
        </p>
        {isUploading && (
          <div className="mt-6 w-48 h-1 bg-gray-200 rounded-full overflow-hidden">
            <div className="h-full bg-blue-500 rounded-full animate-progress"></div>
          </div>
        )}
        {uploadError && (
          <div className="mt-4 flex items-center text-red-500 bg-red-50 px-4 py-2 rounded-lg">
            <AlertCircle className="w-5 h-5 mr-2" />
            <p className="text-sm">{uploadError}</p>
          </div>
        )}
      </div>
    </div>
  );
}