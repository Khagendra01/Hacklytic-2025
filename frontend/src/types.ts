export interface VideoAnalysis {
  masked_video_url: string;
  processed_video_url: string;
  status: 'loading' | 'processing' | 'completed';
  shot_metrics: Array<{
    elbow_angle: number;
    wrist_angle: number;
    shoulder_angle: number;
    knee_angle: number;
    shot_trajectory: number;
    release_height_ratio: number;
    player_height_pixels: number;
    torso_size_pixels: number;
    torso_ratio: number;
    release_height: number;
    ideal_ranges: {
      elbow_angle: [number, number];
      wrist_angle: [number, number];
      shoulder_angle: [number, number];
      knee_angle: [number, number];
      shot_trajectory: [number, number];
      release_height_ratio: [number, number];
      torso_ratio: [number, number];
    };
  }>;
  analysis: {
    coach_recommendations: string;
    coach_reasoning: string;
  };
}