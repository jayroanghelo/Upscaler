# 🚀 ProUpscaler v3.0

Professional image upscaling and restoration suite powered by **FastAPI**, **Spandrel**, and **Gemini AI**.

ProUpscaler provides a state-of-the-art interface and backend for high-quality image enhancement, specializing in both general upscaling (RealESRGAN) and advanced face restoration (CodeFormer).

## ✨ Features

- **Advanced Face Restoration**: Utilizes CodeFormer for stunning facial detail recovery.
- **High-Quality Upscaling**: Integrated RealESRGAN (x2 and x4) for sharp, realistic image expansion.
- **AI-Powered Analysis**: Integrated Gemini AI to analyze images and recommend the best processing parameters automatically.
- **Tiled Processing**: Efficiently handles large images by processing in tiles to prevent memory (OOM) issues.
- **FastAPI Backend**: High-performance asynchronous API.
- **Modern Web Interface**: Clean, responsive UI for easy management of upscaling jobs.
- **Automated Housekeeping**: Periodic cleanup of temporary files to maintain system health.

## 🛠 Tech Stack

- **Backend**: Python 3.13+, FastAPI, Uvicorn.
- **AI/ML Libraries**: Spandrel, PyTorch, FaceXLib, OpenCV.
- **AI Integration**: Google Generative AI (Gemini).
- **Frontend**: HTML5, Vanilla CSS, Javascript.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher.
- (Optional) CUDA or MPS compatible hardware for GPU acceleration.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jayroanghelo/Upscaler.git
   cd Upscaler
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Models**:
   Run the helper script to fetch the necessary pre-trained weights:
   ```bash
   python download_models.py
   ```

### Running the Application

Start the server using the provided shell script or directly via Python:

```bash
./run.sh
```

The application will be available at `http://localhost:8765`.

## 📂 Project Structure

- `app.py`: Main FastAPI application logic and API endpoints.
- `static/`: Frontend assets (HTML, CSS, JS).
- `models/`: Pre-trained model weights (stored locally).
- `uploads/`: Temporary storage for uploaded images.
- `output/`: Processed images ready for download.
- `thumbs/`: Low-resolution previews for the UI.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
Developed by [Jayro Anghelo](https://github.com/jayroanghelo)
