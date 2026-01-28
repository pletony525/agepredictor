# 👤 Age Detection Web App

A web application that detects faces in images and predicts their age using deep learning models.

## 📋 Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## 🚀 Quick Start

### 1. Clone or Download this Directory

Make sure you have all the files in this folder.

### 2. Install Required Packages

Open a terminal/command prompt in this directory and run:

```bash
pip install opencv-python numpy streamlit pillow
```

### 3. Verify Model Files

Make sure these model files are in the same directory:
- `opencv_face_detector.pbtxt`
- `opencv_face_detector_uint8.pb`
- `age_deploy.prototxt`
- `age_net.caffemodel`

**If `age_net.caffemodel` is missing**, download it by running:

```bash
# On Windows PowerShell:
Invoke-WebRequest -Uri "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel" -OutFile "age_net.caffemodel"

# On Mac/Linux:
wget https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel
```

### 4. Run the App

```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

## 📖 How to Use

1. Click on **"Browse files"** to upload an image
2. The app will automatically detect faces and predict ages
3. View results:
   - Original image on the left
   - Processed image with age predictions on the right
   - Green boxes highlight detected faces
   - Yellow text shows predicted age ranges

## 🎯 Supported Image Formats

- JPG / JPEG
- PNG

## 📦 Dependencies

- `opencv-python` - Computer vision and deep learning
- `numpy` - Numerical computing
- `streamlit` - Web app framework
- `pillow` - Image processing

## 🛑 Stopping the App

Press `Ctrl + C` in the terminal where the app is running, or close the terminal window.

## 🔧 Troubleshooting

**Problem:** "Module not found" error  
**Solution:** Make sure all packages are installed: `pip install opencv-python numpy streamlit pillow`

**Problem:** "Can't open model file" error  
**Solution:** Verify all `.pb`, `.pbtxt`, `.prototxt`, and `.caffemodel` files are in the directory

**Problem:** No faces detected  
**Solution:** Try images with clear, front-facing faces with good lighting

## 📝 Age Categories

The model predicts ages in these ranges:
- (0-2) years
- (4-6) years
- (8-12) years
- (15-20) years
- (25-32) years
- (38-43) years
- (48-53) years
- (60-100) years

## 💡 Tips

- Use images with clear, well-lit faces for best results
- The model works best with front-facing faces
- Multiple faces in one image are supported

---

Built with OpenCV and Streamlit
