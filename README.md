# Age Detection Web App

## Setup & Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Model File

```bash
# Windows PowerShell:
Invoke-WebRequest -Uri "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel" -OutFile "age_net.caffemodel"

# Mac/Linux:
wget https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`
