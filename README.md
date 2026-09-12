# Car_Plate_Detection
# 🚘 Advanced ANPR & Vehicle Targeting Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)](https://github.com/ultralytics/ultralytics)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-GPU-orange.svg)](https://github.com/JaidedAI/EasyOCR)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green.svg)](https://opencv.org/)

A high-performance, real-time Computer Vision pipeline designed to track, filter, and identify a specific target vehicle in video streams based on its **paint color** and **license plate number**. 

Unlike standard ANPR systems that blindly run heavy OCR on every car, this pipeline uses a smart "gatekeeper" architecture. It filters out vehicles of the wrong color mathematically before they ever reach the OCR engine, saving massive amounts of GPU compute and allowing for real-time video processing.

## ✨ Key Features
* **Smart Color Gating:** Uses **CLAHE** (Contrast Limited Adaptive Histogram Equalization) and **K-Means Clustering** to find the dominant vehicle color. It calculates distance in the **HSV color space** to ignore shadows, sun glare, and lighting changes.
* **Temporal Tracking:** Uses **Deep SORT** to track vehicles across frames, ensuring the system remembers cars and updates their best OCR read over time.
* **GPU-Accelerated OCR:** Swapped legacy Tesseract for **EasyOCR** running on PyTorch CUDA for highly accurate, deep-learning-based text recognition.
* **Fuzzy String Matching:** Uses SequenceMatcher to handle slight OCR misreads (e.g., confusing a `0` for an `O`), ranking the best potential matches.
* **Memory & Performance Protected:** Implements frame-skipping intervals and resizes cached frames to prevent Out-Of-Memory (OOM) crashes on long video files.

---

## 🧠 How the Pipeline Works

1. **Vehicle Detection:** `YOLOv8n` detects cars, trucks, and motorbikes in the current frame.
2. **Tracking:** `Deep SORT` assigns a unique ID to the vehicle and tracks it across time.
3. **Color Filtering:** The vehicle crop is contrast-normalized and clustered. If its dominant HSV color doesn't match the target HEX color (within a strict threshold), the vehicle is ignored.
4. **Plate Detection:** If the color matches, a secondary custom `YOLOv8` model locates the license plate.
5. **Text Recognition:** `EasyOCR` reads the raw text from the plate crop.
6. **Scoring:** The text is cleaned and fuzzy-matched against the target plate string. The highest-scoring frame for each vehicle is saved.

---
## 📊 Dataset & Models
The custom YOLOv8 license plate detection model (`best.pt`) was trained using the [Car License Plate Detection dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) provided by Andrewmvd on Kaggle.

This dataset contains 433 images with bounding box annotations for car license plates, which provides a strong baseline for the plate detection phase of this pipeline.
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)
## ⚙️ Installation

### 1. Clone the repository
```bash
git clone [https://github.com/Saijosh/Car_Plate_Detection.git](https://github.com/Saijosh/Car_Plate_Detection.git)
cd Car_Plate_Detection
# Install PyTorch with CUDA support (adjust for your CUDA version if needed)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# Install core libraries
pip install ultralytics deep_sort_realtime easyocr

# Install stable Numpy 1.x (to prevent OpenCV incompatibility)
pip install "numpy<2"

# Ensure the desktop version of OpenCV is installed (removes headless conflicts)
pip uninstall opencv-python-headless -y
pip install opencv-python==4.10.0.84
