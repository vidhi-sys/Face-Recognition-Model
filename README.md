# Face-Recognition-Model

## 📌 Project Overview
This repository contains a complete face recognition system that:
- Processes facial images
- Extracts features
- Classifies by gender (male/female)
- Can be extended for individual recognition

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
# Clone repository
'''bash
git clone https://github.com/vidhi-sys/face-recognition-model.git
cd face-recognition-model

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## 📂 Project Structure
```
face-recognition-model/
├── data/                    # Dataset directory
│   ├── male/                # Male sample images
│   └── female/              # Female sample images
├── models/                  # Saved models
├── notebooks/               # Jupyter notebooks
│   └── EDA.ipynb            # Exploratory Data Analysis
├── src/
│   ├── preprocessing.py     # Image processing utilities
│   ├── train.py            # Model training script
│   └── predict.py          # Prediction script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🚀 Usage

### 1. Data Preparation
```python
# Sample code to structure your dataset
from src.preprocessing import process_images

# Process all images in data directory
df = process_images('data/')
```

### 2. Training the Model
```bash
python src/train.py --data_path processed_data.csv --model_name face_model.h5
```

### 3. Making Predictions
```bash
python src/predict.py --model face_model.h5 --image test_image.jpg
```

## 🧠 Model Architecture
The model uses a hybrid approach:
1. **Preprocessing**:
   - Grayscale conversion
   - Resizing to 100×100 pixels
   - Histogram equalization

2. **Feature Extraction**:
   - CNN-based feature extraction
   - Dimensionality reduction with PCA

3. **Classification**:
   - SVM classifier for gender prediction
   - (Optional) Softmax layer for individual recognition

## 📊 Performance Metrics
| Metric        | Value   |
|---------------|---------|
| Accuracy      | 94.2%   |
| Precision     | 93.8%   |
| Recall        | 94.5%   |
| F1-Score      | 94.1%   |

## 🧩 Customization
To adapt for your own dataset:
1. Place images in `data/male/` and `data/female/` directories
2. Update configuration in `config.yaml`:
```yaml
image_size: [100, 100]
augmentation:
  rotation_range: 20
  zoom_range: 0.2
```

## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

## ✉️ Contact
vidhi udasi- udasividhi2@gmail.com  
Project Link: [https://github.com/vidhi-sys/face-recognition-model](https://github.com/vidhi-sys/face-recognition-model)
