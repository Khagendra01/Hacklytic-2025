export interface VideoAnalysis {
  maskedUrl: string;
  processedUrl: string;
  status: 'loading' | 'processing' | 'completed';
  shotMetrics: any[];
}