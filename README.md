# Hacklytic 2025 - Basketball Shot Analysis Platform

A comprehensive AI-powered basketball shot analysis platform that uses computer vision and machine learning to analyze shooting form, provide coaching feedback, and track player improvement over time.

![Basketball Analysis](backend/demo.gif)

## 🏀 Overview

Hacklytic 2025 (Scout It) is an advanced basketball analytics platform that combines computer vision, pose estimation, and AI to provide detailed shot form analysis. The platform helps players improve their shooting mechanics by analyzing key metrics like release angle, follow-through, and shot trajectory.

## ✨ Key Features

### 🎯 Shot Form Analysis
- **Pose Estimation**: Uses MediaPipe to detect and track player body joints
- **Release Angle Measurement**: Calculates optimal release angles for free throws
- **Follow-Through Detection**: Analyzes whether players maintain proper form after release
- **Shot Trajectory Analysis**: Measures arc and angle of the ball's path

### 🤖 AI-Powered Coaching
- **Quick Analysis Agent**: Provides instant feedback on shooting form using Google Gemini Vision
- **Talent Scout Agent**: Compares player performance across multiple videos to track improvement
- **Personalized Recommendations**: AI-generated coaching tips based on detected metrics
- **Visual Analysis**: Combines video analysis with metric data for comprehensive feedback

### 📊 Advanced Metrics
- Elbow angle (ideal: 165-175°)
- Wrist angle (ideal: 70-90°)
- Shoulder angle (ideal: 90-110°)
- Knee angle (ideal: 140-170°)
- Shot trajectory (ideal: 45-55°)
- Release height ratio
- Torso ratio

### 🎥 Video Processing
- **Noise Reduction**: Preprocessing pipeline to improve video quality
- **Basketball Detection**: YOLOv8 model trained to detect basketballs and hoops
- **Annotated Output**: Processed videos with visual overlays showing key metrics
- **Firebase Storage**: Cloud storage integration for video management

## 🏗️ Architecture

### Backend (`/backend`)
- **FastAPI**: RESTful API server
- **Computer Vision**: OpenCV, MediaPipe, YOLOv8
- **AI Agents**: LangChain, Google Gemini, LangGraph
- **Video Processing**: Custom noise masking and shot detection pipeline
- **Storage**: Firebase Storage integration

### Frontend (`/frontend`)
- **React + TypeScript**: Modern web interface
- **Vite**: Fast build tool and dev server
- **Tailwind CSS**: Utility-first styling
- **Firebase**: Client-side storage integration

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- Firebase account and credentials
- Google API key for Gemini

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the `backend` directory:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   FIREBASE_CREDENTIALS_PATH=path/to/firebase-credentials.json
   ```

5. **Download YOLO weights**:
   - Ensure `runs/detect/train/weights/best.pt` exists
   - Or download pre-trained weights to `Yolo-Weights/` directory

6. **Run the server**:
   ```bash
   uvicorn main:app --reload
   ```

   The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Configure Firebase**:
   Update `src/firebaseConfig.ts` with your Firebase configuration:
   ```typescript
   const firebaseConfig = {
     apiKey: "your-api-key",
     authDomain: "your-auth-domain",
     projectId: "your-project-id",
     storageBucket: "your-storage-bucket",
     messagingSenderId: "your-sender-id",
     appId: "your-app-id"
   };
   ```

4. **Update API endpoint** (if needed):
   Update `src/api.ts` with your backend URL

5. **Run development server**:
   ```bash
   npm run dev
   ```

   The frontend will be available at `http://localhost:5173`

## 📖 Usage

### Basic Shot Analysis

1. **Upload a video** through the web interface
2. The system will:
   - Preprocess the video (noise reduction)
   - Detect the basketball and player pose
   - Extract shooting metrics
   - Generate AI-powered analysis
3. **View results** including:
   - Annotated video with pose overlays
   - Detailed metrics breakdown
   - Coaching recommendations

### Pro Analysis (Talent Scout)

1. Navigate to the **Pro** page
2. Upload two videos (initial and follow-up)
3. Specify the time period between videos
4. Get comprehensive analysis including:
   - Initial and follow-up assessments
   - Improvement tracking
   - Potential ceiling projection
   - Tier ranking (S, A, B, C)

## 🔧 API Endpoints

### `POST /api/mask_video`
Process a single video for shot analysis.

**Request Body**:
```json
{
  "video_url": "https://example.com/video.mp4"
}
```

**Response**:
```json
{
  "masked_video_url": "https://firebase.../masked_video.mp4",
  "processed_video_url": "https://firebase.../processed_video.mp4",
  "shot_metrics": [...],
  "analysis": {
    "coach_recommendations": "...",
    "coach_reasoning": "..."
  }
}
```

### `POST /api/talent_scout`
Compare two videos for improvement analysis.

**Request Body**:
```json
{
  "video1_url": "https://example.com/video1.mp4",
  "video2_url": "https://example.com/video2.mp4"
}
```

**Response**:
```json
{
  "status": "success",
  "video1_url": "...",
  "video2_url": "...",
  "video1_metrics": {...},
  "video2_metrics": {...},
  "analysis": {
    "initial_assessment": {...},
    "follow_up_assessment": {...},
    "improvement_analysis": {...},
    "potential_assessment": {...}
  }
}
```

## 🧪 Training Custom Models

### YOLO Model Training

The project includes a custom YOLOv8 model trained on basketball detection. To retrain:

1. **Prepare dataset**:
   - Organize images in `basketball.v1i.yolov8/` directory
   - Ensure proper train/val/test splits

2. **Update config**:
   - Edit `config.yaml` with your dataset paths

3. **Train model**:
   ```bash
   python train_model.py
   ```

4. **Update model path**:
   - Update `shot_detector.py` to use your trained weights

## 📁 Project Structure

```
Hacklytic-2025/
├── backend/
│   ├── agents/              # AI analysis agents
│   │   ├── quick_analysis_agent.py
│   │   ├── talent_scout_agent.py
│   │   └── ...
│   ├── firebase/            # Firebase integration
│   ├── preprocessing/       # Video preprocessing
│   ├── basketball.v1i.yolov8/  # YOLO dataset
│   ├── main.py              # FastAPI server
│   ├── shot_detector.py     # Core shot detection logic
│   ├── requirements.txt     # Python dependencies
│   └── config.yaml          # Configuration file
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── pages/           # Page components
│   │   ├── api.ts           # API client
│   │   └── ...
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

## 🛠️ Technologies Used

### Backend

#### Web Framework & API
- **FastAPI** (0.115.8): Modern, fast Python web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI applications
- **Pydantic**: Data validation using Python type annotations
- **Starlette**: Lightweight ASGI framework (FastAPI dependency)

#### Computer Vision & Image Processing
- **OpenCV** (4.8.0.74): Computer vision library for image/video processing
- **opencv-contrib-python**: Extended OpenCV functionality
- **MediaPipe** (0.10.20): Google's framework for pose estimation and body tracking
- **cvzone** (1.5.6): Computer vision utilities built on OpenCV
- **Pillow** (10.0.0): Image processing library
- **scikit-image** (0.19.3): Image processing algorithms
- **imageio** (2.31.1): Video and image I/O

#### Machine Learning & AI
- **Ultralytics YOLOv8** (8.0.145): State-of-the-art object detection model
- **PyTorch** (2.0.1): Deep learning framework
- **torchvision** (0.15.2): Computer vision utilities for PyTorch
- **google-generativeai**: Google Gemini AI SDK for video and text analysis
- **LangChain**: Framework for building LLM applications
- **langchain-core**: Core LangChain functionality
- **langchain-openai**: OpenAI integration for LangChain
- **LangGraph**: Graph-based agent orchestration

#### Scientific Computing
- **NumPy** (≥1.24.0): Numerical computing library
- **SciPy** (1.11.1): Scientific computing library
- **Pandas** (2.0.3): Data manipulation and analysis
- **matplotlib** (3.7.2): Plotting and visualization
- **scikit-image**: Image processing algorithms

#### Video Processing
- **filterpy** (1.4.5): Kalman filtering and tracking
- **PyWavelets** (1.4.1): Wavelet transforms for signal processing

#### Cloud & Storage
- **firebase-admin**: Firebase Admin SDK for server-side operations
- **google-auth** (2.22.0): Google authentication library
- **google-auth-oauthlib** (1.0.0): OAuth2 support for Google APIs

#### Utilities
- **python-dotenv** (1.0.1): Environment variable management
- **aiohttp**: Async HTTP client/server
- **httpx** (0.28.1): Modern HTTP client
- **pymongo** (4.11.1): MongoDB driver (if used for data storage)
- **tqdm** (4.65.0): Progress bars for loops
- **python-multipart**: Multipart form data parsing

### Frontend

#### Core Framework
- **React** (18.3.1): JavaScript library for building user interfaces
- **React DOM** (18.3.1): React renderer for web
- **TypeScript** (5.5.3): Typed superset of JavaScript

#### Build Tools & Development
- **Vite** (5.4.2): Next-generation frontend build tool
- **@vitejs/plugin-react** (4.3.1): React plugin for Vite
- **ESLint** (9.9.1): JavaScript linter
- **TypeScript ESLint** (8.3.0): TypeScript-specific linting rules

#### Routing
- **react-router-dom** (7.2.0): Declarative routing for React

#### Styling
- **Tailwind CSS** (3.4.1): Utility-first CSS framework
- **PostCSS** (8.4.35): CSS transformation tool
- **Autoprefixer** (10.4.18): CSS vendor prefixing
- **@tailwindcss/typography** (0.5.16): Typography plugin for Tailwind

#### Cloud Services
- **Firebase** (11.3.1): Google's platform for web and mobile apps
  - Firebase Storage for video/file storage
  - Firebase Authentication (if used)

#### UI Utilities
- **lucide-react** (0.344.0): Icon library
- **clsx** (2.1.0): Utility for constructing className strings conditionally

## 🎯 Ideal Shooting Metrics

The system evaluates shots based on these ideal ranges:

| Metric | Ideal Range | Description |
|--------|-------------|-------------|
| Elbow Angle | 165-175° | Arm extension at release |
| Wrist Angle | 70-90° | Wrist snap position |
| Shoulder Angle | 90-110° | Shoulder alignment |
| Knee Angle | 140-170° | Leg bend for power |
| Shot Trajectory | 45-55° | Optimal arc angle |
| Release Height Ratio | 1.8-2.3 | Height relative to body |
| Torso Ratio | 0.3-0.4 | Body proportions |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is part of Hacklytic 2025. All rights reserved.

## 🙏 Acknowledgments

- MediaPipe team for pose estimation models
- Ultralytics for YOLOv8
- Google for Gemini AI
- OpenCV community

## 📧 Contact

For questions or support, please open an issue on the repository.

---

**Built with ❤️ for basketball players and coaches**

