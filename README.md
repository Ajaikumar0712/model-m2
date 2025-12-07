# 🌱 Plant Stress Detection System

A real-time plant stress detection application using computer vision and vegetation indices. This system analyzes plant health through image processing and provides visual feedback via heatmaps and YOLO-style annotations.

## ✨ Features

- **📂 Image Upload Mode**: Analyze static plant images
- **📹 Live Camera Feed**: Real-time plant stress detection using webcam
- **🔥 Stress Heatmap**: Visual representation of plant stress levels
- **📍 YOLO Annotations**: Automatic detection and annotation of stressed regions
- **⚙️ Customizable Parameters**: Adjust detection thresholds and weights in real-time
- **📊 JSON Export**: Export YOLO-format segmentation annotations

## 🎯 Detection Methods

The system uses multiple vegetation indices to detect plant stress:

- **ExG (Excess Green Index)**: Highlights green vegetation
- **VARI (Visible Atmospherically Resistant Index)**: Reduces atmospheric effects
- **HSV Analysis**: Hue, Saturation, Value color space analysis
- **Yellow Proximity Detection**: Identifies yellowing/stressed leaves

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Webcam (for live feed mode)
- Windows/Linux/MacOS

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/model-m2.git
   cd model-m2
   ```

2. **Create a virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux/Mac
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   
   If using `uv` (recommended):
   ```bash
   uv sync
   ```
   
   Or using pip:
   ```bash
   pip install opencv-python pillow streamlit ultralytics
   ```

### Running the Application

1. **Activate the virtual environment** (if not already activated)
   ```bash
   # Windows
   .venv\Scripts\activate

   # Linux/Mac
   source .venv/bin/activate
   ```

2. **Start the Streamlit app**
   ```bash
   # Windows
   python -m streamlit run model.py

   # Linux/Mac  
   streamlit run model.py
   ```

3. **Open your browser**
   
   The app will automatically open at `http://localhost:8501`

## 📖 Usage Guide

### Upload Image Mode

1. Navigate to the **"📂 Upload Image"** tab
2. Click "Browse files" or drag & drop an image
3. Supported formats: JPG, JPEG, PNG
4. Adjust parameters in the sidebar as needed
5. View results:
   - Original image
   - Stress heatmap overlay
   - Annotated stress regions
   - YOLO segmentation data (JSON)

### Live Camera Feed Mode

1. Navigate to the **"📹 Live Camera Feed"** tab
2. Click **"▶️ Start"** to activate your webcam
3. Grant camera permissions when prompted
4. View real-time analysis in three panels:
   - 🎥 Live camera feed
   - 🔥 Stress heatmap
   - 📍 Annotated regions
5. Click **"⏹️ Stop"** to end the feed
6. Adjust detection parameters on-the-fly using the sidebar

## ⚙️ Parameters

### Thresholds

- **ExG Threshold** (0-100): Minimum ExG value for vegetation detection
  - Default: 45
  - Lower = More sensitive, may include non-vegetation
  - Higher = More strict, may miss some vegetation

- **VARI Threshold** (0-100): Minimum VARI value for vegetation detection
  - Default: 50
  - Similar sensitivity trade-off as ExG

- **Stress Cutoff** (0-100): Minimum stress score to flag a region
  - Default: 50
  - Lower = Detect mild stress
  - Higher = Only detect severe stress

- **Minimum Area** (100-10000 px): Minimum pixel area for stress region detection
  - Default: 2000
  - Filters out small noise/artifacts

### Weights

Fine-tune the stress calculation formula:

- **Weight ExG** (0-100): Influence of ExG inverseon stress score (default: 35)
- **Weight VARI** (0-100): Influence of VARI inverse on stress score (default: 30)
- **Weight Yellow** (0-100): Influence of yellowing on stress score (default: 25)
- **Weight Sat** (0-100): Influence of low saturation on stress score (default: 10)

## 📁 Project Structure

```
model-m2/
├── model.py              # Main Streamlit application
├── test.py               # Testing utilities
├── main.py               # Additional scripts
├── pyproject.toml        # Project dependencies
├── uv.lock              # Dependency lock file
├── .venv/               # Virtual environment
├── img/                 # Sample images
├── best.pt              # Trained model weights
├── trained_model.pt     # Alternative model weights
├── yolov8n.pt          # YOLO base model
└── README.md           # This file
```

## 🛠️ Troubleshooting

### Camera Not Working

**Problem**: "Cannot access camera" error

**Solutions**:
- Ensure your webcam is connected and not used by another application
- Check camera permissions in your browser/system settings
- Try a different browser (Chrome/Edge recommended)
- Restart the application

### Module Not Found Error

**Problem**: `ModuleNotFoundError: No module named 'cv2'`

**Solutions**:
```bash
# Make sure virtual environment is activated
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Reinstall dependencies
uv sync
# or
pip install opencv-python pillow streamlit
```

### "Failed to canonicalize script path"

**Problem**: Error when running `streamlit run model.py`

**Solution**:
```bash
# Use python -m instead
python -m streamlit run model.py
```

### Slow Performance

**Solutions**:
- Reduce camera resolution
- Increase the `time.sleep()` value in the live feed loop (line ~215)
- Lower the minimum area threshold to process fewer regions
- Use a GPU-enabled version of OpenCV if available

## 🔬 Technical Details

### Vegetation Indices

**ExG (Excess Green)**:
```
ExG = 2G - R - B
```

**VARI (Visible Atmospherically Resistant Index)**:
```
VARI = (G - R) / (G + R - B)
```

### Stress Calculation

```python
stress_score = w_exg * (1 - ExG) + 
               w_var * (1 - VARI) + 
               w_yellow * yellow_proximity + 
               w_sat * (1 - Saturation)
```

## 📝 Requirements

See `pyproject.toml` for full dependency list:

- opencv-python >= 4.12.0
- pillow >= 11.3.0
- streamlit >= 1.50.0
- ultralytics >= 8.3.209
- numpy (installed with opencv)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLO architecture by Ultralytics
- Vegetation indices based on agricultural research
- Streamlit for the interactive web framework

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Happy Plant Monitoring! 🌿**
