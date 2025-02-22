# Avi Shah - Basketball Shot Detector/Tracker - July 2023

import asyncio
import math
import os
import subprocess

import cv2
import cvzone
import numpy as np
import PoseEstimationMin as pm
from preprocessing.noise_masking import NoiseReducer
from ultralytics import YOLO
from utils import (
    clean_ball_pos,
    clean_hoop_pos,
    detect_down,
    detect_up,
    in_hoop_region,
    score,
)


class ShotDetector:
    def __init__(self):
        # Load the YOLO model created from main.py - change text to your relative path
        self.model = YOLO("backend/runs/detect/train/weights/best.pt")
        self.class_names = ['Basketball', 'Basketball Hoop']

        # Uncomment line below to use webcam (I streamed to my iPhone using Iriun Webcam)
        # self.cap = cv2.VideoCapture(0)

        # Use video - replace text with your video path
        self.cap = cv2.VideoCapture('backend/noisy_images/masked_shot.mp4')
        # self.cap = cv2.VideoCapture('shoot-with-great-form (online-video-cutter.com).mp4')
        # self.cap = cv2.VideoCapture('3fe1d32b-813c-4034-8307-aaacacf1c87c.mp4')
        # self.cap = cv2.VideoCapture('shoot-with-great-form (online-video-cutter.com)-2.mp4')


        self.ball_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.hoop_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)

        self.frame_count = 0
        self.frame = None

        self.makes = 0
        self.attempts = 0

        # Used to detect shots (upper and lower region)
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        # Used for green and red colors after make/miss
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        self.detector = pm.poseDetector()
        self.img = None
        self.lmList = []

        self.below_shoulder = False
        self.above_shoulder = False

        # Add new instance variables for storing angles
        self.shot_metrics = []  # Will store metrics for each shot

        # Add video writer setup
        self.output_path = 'output_video.mp4'
        frame = self.cap.read()[1]  # Read first frame to get dimensions
        self.frame_width = int(frame.shape[1])
        self.frame_height = int(frame.shape[0])
        self.out = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            30,  # FPS
            (self.frame_width, self.frame_height)
        )

        # Add video FPS tracking
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.shot_timestamps = []  # Will store timestamps of shots

        try:
            release_angle, follow_through_angle, left_feet_heights, right_feet_heights, shot_metrics = self.run()
            
            print(f'\nShooting Form Analysis:')
            print(f'Shot Trajectory: {release_angle:.1f}° (Ideal: 45-55°)')
            print(f'Follow Through Angle: {follow_through_angle:.1f}°')
            
            # Print detailed metrics for each shot
            for i, metrics in enumerate(shot_metrics):
                print(f"\nShot {i+1} Metrics:")
                ideal = metrics['ideal_ranges']
                
                def print_metric(name, value, ideal_range):
                    status = "GOOD" if ideal_range[0] <= value <= ideal_range[1] else "ADJUST"
                    feedback = ""
                    
                    if status == "ADJUST":
                        if value < ideal_range[0]:
                            if name == "Shot Trajectory":
                                feedback = "- Shot too flat, increase release angle"
                            elif name == "Elbow Angle":
                                feedback = "- Elbow too bent, extend more"
                            elif name == "Wrist Angle":
                                feedback = "- Not enough wrist flexion"
                            elif name == "Shoulder Angle":
                                feedback = "- Arms too low, raise shooting pocket"
                            elif name == "Knee Angle":
                                feedback = "- Bending knees too much"
                        else:
                            if name == "Shot Trajectory":
                                feedback = "- Shot too high, decrease release angle"
                            elif name == "Elbow Angle":
                                feedback = "- Elbow locked, maintain slight bend"
                            elif name == "Wrist Angle":
                                feedback = "- Too much wrist flexion"
                            elif name == "Shoulder Angle":
                                feedback = "- Arms too high, lower shooting pocket"
                            elif name == "Knee Angle":
                                feedback = "- Not enough knee bend"
                    
                    print(f"{name}: {value:.1f}° (Ideal: {ideal_range[0]}-{ideal_range[1]}°) - {status} {feedback}")
                
                print_metric("Elbow Angle", metrics['elbow_angle'], ideal['elbow_angle'])
                print_metric("Wrist Angle", metrics['wrist_angle'], ideal['wrist_angle'])
                print_metric("Shoulder Angle", metrics['shoulder_angle'], ideal['shoulder_angle'])
                print_metric("Knee Angle", metrics['knee_angle'], ideal['knee_angle'])
                print_metric("Shot Trajectory", metrics['shot_trajectory'], ideal['shot_trajectory'])
                
                # Add release height feedback
                ratio = metrics['release_height_ratio']
                print(f"Release Height Ratio: {ratio:.2f}", end=" ")
                if ratio < 1.8:
                    print("- Releasing too low, extend up more")
                elif ratio > 2.3:
                    print("- Releasing too high, might affect consistency")
                else:
                    print("- GOOD")
            
            left_release_height = np.mean(left_feet_heights[0:50]) - left_feet_heights[-1]
            right_release_height = np.mean(right_feet_heights[0:50]) - right_feet_heights[-1]
            print(f'RELEASE HEIGHT: {np.mean([left_release_height, right_release_height]):.1f}')
            
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
        finally:
            # Update cleanup to include video writer
            if hasattr(self, 'cap'):
                self.cap.release()
            if hasattr(self, 'out'):
                self.out.release()
            cv2.destroyAllWindows()

    def run(self):
        release_angle = None
        follow_through_angle = -1
        left_feet_heights = []
        right_feet_heights = []
        last_angle = 0
        num_frame = 0
        angle_decreasing = False
        already_beeped = False
        release_frame = -1
        already_released = False
        follow_through_iter = 10
        follow_through_best = 155
        did_follow_through = False
        shot_metrics = []

        while True:
            ret, self.frame = self.cap.read()
            
            # Check if frame was successfully read
            if not ret or self.frame is None:
                print("\nEnd of video reached")
                break
                
            self.img = self.detector.findPose(self.frame, draw=False)
            self.lmList = self.detector.findPosition(self.img, draw=False)
            
            # Check if pose detection was successful
            if not self.lmList:
                print("\nNo pose detected in frame")
                continue

            self.detector.findAngle(self.img, 12, 14, 16, draw=True)

            results = self.model(self.frame, stream=True)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Class Name
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]

                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    # Only create ball points if high confidence or near hoop
                    if (conf > 0.4 or (in_hoop_region(center, self.hoop_pos) and conf > 0.25)) and current_class == "Basketball":
                        self.ball_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))

                    # Create hoop points if high confidence
                    if conf > 0.6 and current_class == "Basketball Hoop":
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))

            cv2.circle(self.img, (self.lmList[29][1], self.lmList[29][2]), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(self.img, (self.lmList[30][1], self.lmList[30][2]), 10, (0, 0, 255), cv2.FILLED)

            left_feet_heights.append(self.lmList[29][2])
            right_feet_heights.append(self.lmList[30][2])

            # if did_follow_through:
            #     if follow_through_angle > follow_through_best:
            #         cv2.putText(self.frame, 'GOOD FOLLOW THROUGH', (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            #     else:
            #         cv2.putText(self.frame, 'DIDN\'T FOLLOW THROUGH', (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            if already_released and not did_follow_through:
                if self.frame_count - release_frame > follow_through_iter:
                    angle = self.detector.findAngle(self.img, 12, 14, 16, draw=True)

                    if angle > follow_through_angle:
                        follow_through_angle = angle

                    # cv2.imwrite(f'follow-through/{temp}.jpg', self.img)
                    follow_through_iter += 1
                    # print(f'{temp} error -> {follow_through_angle} angle')

                    if follow_through_iter >= 40:
                        did_follow_through = True

                # if self.frame_count - release_frame < 12:
                #     cv2.putText(self.frame, 'REMEMBER TO FOLLOW THROUGH', (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)


            elif not did_follow_through:
                if(self.ball_pos):
                    bx, by = self.ball_pos[-1][0][0], self.ball_pos[-1][0][1] #most recent coordinates for ball
                    sx, sy = self.lmList[12][1], self.lmList[12][2] #shoulder coordinates

                    angle_error = 15
                    error_margin = 50
                    dist_margin = 125

                    if(sy < by):
                        self.below_shoulder = True
                    elif self.below_shoulder and sy > by + error_margin:
                        angle = self.detector.findAngle(self.img, 12, 14, 16, draw=True)

                        if angle < last_angle:
                                angle_decreasing = True
                        else:
                            angle_decreasing = False

                        if self.frame_count % 10 == 0:
                            last_angle = angle

                        # if angle_decreasing and abs(angle - 60) <= angle_error:
                        #     print(f'loaded angle: {angle}')
                        #     cv2.putText(self.frame, 'RELEASE NOW', (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

                            if not already_beeped:
                                subprocess.run('osascript -e "beep"', shell=True)
                                already_beeped = True

                            # cv2.imwrite('loaded_angle.jpg', self.img)

                        self.above_shoulder = True

                        rpx, rpy = self.lmList[12][1], self.lmList[12][2] #right pinky

                        dist = math.dist((rpx, rpy), (bx, by))

                        if dist > dist_margin:
                            # Capture timestamp
                            timestamp = self.frame_count / self.fps  # Convert frames to seconds
                            minutes = int(timestamp // 60)
                            seconds = timestamp % 60
                            self.shot_timestamps.append(f"{minutes}:{seconds:05.2f}")
                            
                            # Capture all important angles at release
                            shoulder_pos = (self.lmList[12][1], self.lmList[12][2])
                            elbow_pos = (self.lmList[14][1], self.lmList[14][2])
                            ball_pos = (bx, by)
                            
                            shot_metric = self.calculate_shot_angles(ball_pos, shoulder_pos, elbow_pos)
                            
                            # Calculate release height relative to player height
                            player_height = self.lmList[24][2] - self.lmList[0][2]  # Distance from hip to nose
                            ball_height = self.lmList[24][2] - by  # Distance from hip to ball
                            shot_metric['release_height_ratio'] = ball_height / player_height if player_height != 0 else 0
                            
                            shot_metrics.append(shot_metric)
                            
                            release_angle = shot_metric['shot_trajectory']
                            release_frame = self.frame_count
                            already_released = True

            self.clean_motion()
            self.shot_detection()
            self.display_score()
            self.frame_count += 1

            # After all drawing is done but before cv2.imshow
            self.out.write(self.frame)
            
            cv2.imshow("Basketball Shot Analysis", self.frame)

            # Close if 'q' is clicked
            if cv2.waitKey(1) & 0xFF == ord('q'):  # higher waitKey slows video down, use 1 for webcam
                break

        # Cleanup
        print("\nFinal Statistics:")
        print(f"Total Shots Attempted: {self.attempts}")
        print(f"Shots Made: {self.makes}")
        if self.attempts > 0:
            print(f"Shooting Percentage: {(self.makes/self.attempts)*100:.1f}%")
        
        self.cap.release()
        cv2.destroyAllWindows()

        if release_angle > 180:
            release_angle = 360 - release_angle

        print("\nShot Release Timestamps:")
        for i, timestamp in enumerate(self.shot_timestamps):
            print(f"Shot {i+1}: {timestamp}")

        return release_angle, follow_through_angle, left_feet_heights, right_feet_heights, shot_metrics

    def clean_motion(self):
        # Clean and display ball motion
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for i in range(0, len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2)

        # Clean hoop motion and display current hoop center
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

    def shot_detection(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            # Add debug visualization
            if len(self.hoop_pos) > 0:
                # Draw detection zones
                hoop = self.hoop_pos[-1]
                x1 = hoop[0][0] - 2 * hoop[2]  # Reduced from 6x
                x2 = hoop[0][0] + 2 * hoop[2]
                y1 = hoop[0][1] - 1.5 * hoop[3]  # Reduced from 4x
                y2 = hoop[0][1]
                
                # Draw detection zones for debugging
                cv2.rectangle(self.frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Up zone
                
                # Draw down detection zone
                y_down = hoop[0][1] + 0.5 * hoop[3]
                cv2.rectangle(self.frame, (int(x1), int(y_down)), (int(x2), int(y_down + hoop[3])), (0, 255, 0), 2)  # Down zone

            # Rest of the detection logic...
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]
                    print(f"\nBall detected in UP area at frame {self.up_frame}")
                    # Reset down detection when new up is detected
                    self.down = False
                    self.down_frame = 0

            # Add minimum frame difference requirement
            if self.up and not self.down and (self.frame_count - self.up_frame) > 5:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]
                    print(f"Ball detected in DOWN area at frame {self.down_frame}")

            # Add minimum frame difference check
            if (self.up and self.down and 
                self.up_frame < self.down_frame and 
                (self.down_frame - self.up_frame) > 10):  # Minimum frames between up and down
                
                print("\nShot Detected!")
                print(f"Up frame: {self.up_frame}")
                print(f"Down frame: {self.down_frame}")
                print(f"Frame difference: {self.down_frame - self.up_frame}")
                
                self.attempts += 1
                self.up = False
                self.down = False

                # If it is a make, put a green overlay
                if score(self.ball_pos, self.hoop_pos):
                    self.makes += 1
                    print("Shot made!")
                    self.overlay_color = (0, 255, 0)
                    self.fade_counter = self.fade_frames
                # If it is a miss, put a red overlay
                else:
                    print("Shot missed!")
                    self.overlay_color = (0, 0, 255)
                    self.fade_counter = self.fade_frames

    def display_score(self):
        # Add text
        # text = str(self.makes) + " / " + str(self.attempts)
        # cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        # cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        # Gradually fade out color after shot
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, np.full_like(self.frame, self.overlay_color), alpha, 0)
            self.fade_counter -= 1

    def calculate_shot_angles(self, ball_pos, shoulder_pos, elbow_pos):
        shot_metric = {}
        
        # Initialize all metrics with default values
        shot_metric.update({
            'elbow_angle': 0,
            'wrist_angle': 0,
            'shoulder_angle': 0,
            'knee_angle': 0,
            'shot_trajectory': 0,
            'release_height_ratio': 0,
            'player_height_pixels': 0,
            'torso_size_pixels': 0,
            'torso_ratio': 0,
            'release_height': 0
        })
        
        # Normalize angles to be within 0-180 degrees
        def normalize_angle(angle):
            if angle is None:
                return 0
            try:
                while angle > 180:
                    angle = 360 - angle
                return abs(angle)
            except:
                return 0
        
        try:
            # Calculate shot trajectory relative to horizontal
            if None not in [ball_pos, shoulder_pos]:
                dx = ball_pos[0] - shoulder_pos[0]
                dy = shoulder_pos[1] - ball_pos[1]  # Flip y because image coordinates
                shot_angle = math.degrees(math.atan2(dy, dx))
                shot_metric['shot_trajectory'] = 90 - shot_angle  # Convert to standard coordinate system
        except Exception as e:
            print(f"Warning: Could not calculate shot trajectory - {str(e)}")
        
        try:
            # Right arm angles
            if self.img is not None and self.detector is not None:
                elbow_angle = self.detector.findAngle(self.img, 12, 14, 16, draw=False)
                wrist_angle = self.detector.findAngle(self.img, 14, 16, 20, draw=False)
                shoulder_angle = self.detector.findAngle(self.img, 24, 12, 14, draw=False)
                knee_angle = self.detector.findAngle(self.img, 24, 26, 28, draw=False)
                
                shot_metric.update({
                    'elbow_angle': normalize_angle(elbow_angle),
                    'wrist_angle': normalize_angle(wrist_angle),
                    'shoulder_angle': normalize_angle(shoulder_angle),
                    'knee_angle': normalize_angle(knee_angle)
                })
        except Exception as e:
            print(f"Warning: Could not calculate joint angles - {str(e)}")
        
        try:
            # Calculate player measurements if landmarks exist
            if self.lmList and len(self.lmList) > 30:  # Check if we have enough landmarks
                foot_y = max(self.lmList[29][2], self.lmList[30][2])  # Use the lower foot position
                head_y = self.lmList[0][2]       # Nose position
                hip_y = self.lmList[24][2]       # Hip position
                shoulder_y = self.lmList[12][2]   # Shoulder position
                
                # Calculate heights (remember y increases downward in image coordinates)
                total_height = abs(foot_y - head_y)
                torso_size = abs(hip_y - shoulder_y)
                
                # Calculate release height (relative to feet)
                if ball_pos is not None:
                    ball_height = abs(foot_y - ball_pos[1])
                    shot_metric.update({
                        'release_height_ratio': ball_height / total_height if total_height != 0 else 0,
                        'release_height': ball_height
                    })
                
                shot_metric.update({
                    'player_height_pixels': total_height,
                    'torso_size_pixels': torso_size,
                    'torso_ratio': torso_size / total_height if total_height != 0 else 0
                })
        except Exception as e:
            print(f"Warning: Could not calculate height measurements - {str(e)}")
        
        # Add ideal ranges for feedback
        shot_metric['ideal_ranges'] = {
            'elbow_angle': (165, 175),
            'wrist_angle': (70, 90),
            'shoulder_angle': (90, 110),
            'knee_angle': (140, 170),
            'shot_trajectory': (45, 55),
            'release_height_ratio': (1.8, 2.3),
            'torso_ratio': (0.3, 0.4)
        }
        
        # Debug prints with safe access
        print("\nDebug Measurements:")
        print(f"Angles detected:")
        print(f"- Elbow: {shot_metric['elbow_angle']}")
        print(f"- Wrist: {shot_metric['wrist_angle']}")
        print(f"- Shoulder: {shot_metric['shoulder_angle']}")
        print(f"- Knee: {shot_metric['knee_angle']}")
        print(f"- Shot trajectory: {shot_metric['shot_trajectory']}")
        print(f"Heights detected:")
        print(f"- Player height: {shot_metric['player_height_pixels']} pixels")
        print(f"- Release height ratio: {shot_metric['release_height_ratio']}")
        return shot_metric

if __name__ == "__main__":
    import asyncio
    import os

    from preprocessing.noise_masking import NoiseReducer

    # Get absolute path to ShotFormCorrector directory
    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # print(f"Project root: {project_root}")  # Debug print
    # Construct full paths using project root
    input_video = "backend/noisy_images/clip_001.mp4"
    output_video = "backend/noisy_images/masked_shot.mp4"
    
    # Debug prints
    print(f"Looking for input video at: {input_video}")
    print(f"Output will be saved to: {output_video}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    
    # Check if input file exists
    if not os.path.exists(input_video):
        print(f"Error: Input video file not found at: {input_video}")
        raise FileNotFoundError(f"Input video not found: {input_video}")
    
    noise_reducer = NoiseReducer()
    
    # Run noise reduction asynchronously and wait for it to complete
    asyncio.run(noise_reducer.process_video(input_video, output_video))
    
    print("MASKED SHOT CREATED")
    
    # Now that noise reduction is complete, run shot detection
    ShotDetector()

