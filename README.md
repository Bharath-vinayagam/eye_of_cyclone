# 🌪️ Cyclone Eye Center Detection

This repository implements and compares deep learning models for detecting the exact center (eye) of a cyclone from multi-spectral satellite imagery. The project utilizes a custom **ResNet50-based Regression Network** (TensorFlow/Keras) and compares it with **YOLOv8** (Ultralytics object detection) to locate the coordinates of cyclone eyes.

---

## 📂 Project Structure

```directory
cyclone_eye_detection/
│
├── cyclone_eye_detection.ipynb  # Core Jupyter notebook containing data pipeline & ResNet50 model
├── requirements.txt            # Python dependencies
├── GITHUB_SETUP.md             # Git setup and deployment guide
│
├── cyclone_dataset.h5          # Raw HDF5 dataset containing satellite images and eye masks
├── cyclone_coco.yaml           # YOLOv8 dataset configuration file
│
├── best_resnet50_model.h5      # Saved best-performing weights for the ResNet50 regression model
├── model_parameters.json       # Hyperparameters and architecture specs for the ResNet50 model
├── model_evaluation_metrics.json # Performance metrics calculated on the validation set
│
├── datasets/                   # Processed dataset formatted for YOLOv8 (images & text labels)
│   └── cyclone_eye/
│       ├── train/              # Training set (images & labels)
│       └── valid/              # Validation set (images & labels)
│
└── runs/                       # YOLOv8 training outputs
    └── train/
        └── cyclone_eye_yolov8/
            ├── weights/        # Trained YOLOv8 model weights (best.pt, last.pt)
            ├── results.csv     # Training loss and evaluation metrics per epoch
            ├── results.png     # Plots showing training losses and metrics over epochs
            └── *.png           # Confusion matrices, F1, PR, Precision, and Recall curves
```

---

## 📊 Evaluation & Metrics Comparison

The project trains and evaluates two distinct approaches:
1. **ResNet50 Custom Regression Head**: Direct coordinate regression optimizing a continuous distance function.
2. **YOLOv8 Object Detector**: Bounding box prediction with a tiny $0.05 \times 0.05$ anchor box centered on the cyclone eye.

### 1. ResNet50 Custom Regression Model Metrics
The regression model maps inputs to a normalized coordinate pair $(x, y) \in [0, 1]^2$. The following metrics were obtained on the validation dataset (image dimensions: $201 \times 201$ pixels):

#### 📐 Regression Metrics (Distance in Pixels)
| Metric | Value (Normalized) | Value (Pixels) | Description |
| :--- | :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | 0.0988 | **19.86 px** | Mean distance in each coordinate axis |
| **Root Mean Squared Error (RMSE)** | - | **25.40 px** | Heavy penalty on outliers |
| **Mean Squared Error (MSE)** | 0.0160 | **645.26 px²** | Variance of coordinate prediction error |
| **Max Error** | - | **112.12 px** | Maximum distance error on validation samples |
| **R² Score** | **-0.173** | - | Comparison against mean coordinate baseline |

#### 🎯 Coordinate Distance Detection Thresholds
If we count a prediction as a "hit" when the predicted coordinate is within $N$ pixels of the true eye center (Euclidean distance):

| Distance Threshold | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Within 1 Pixel** | 0.33% | 0.003 | 1.0 | 0.007 |
| **Within 5 Pixels** | 2.29% | 0.023 | 1.0 | 0.045 |
| **Within 10 Pixels** | 9.06% | 0.091 | 1.0 | 0.166 |
| **Within 20 Pixels** | 30.79% | 0.308 | 1.0 | 0.471 |

*Note: Recall is $1.0$ because the regression head outputs exactly one prediction coordinate per image, and there are no true-negative cases (every sample has an eye).*

#### 📈 Training History (ResNet50)
- **Total Training Epochs**: 16 epochs (Early stopping patience = 15 triggered after validation loss did not improve beyond epoch 1).
- **Best Validation Loss**: **0.0059** (Huber Loss) at Epoch 1 (restored best weights).
- **Validation Distance Error Stats**:
  - **Mean Distance Error**: 30.96 pixels
  - **Median Distance Error**: 27.89 pixels
  - **95th Percentile Error**: 66.07 pixels
  - **99th Percentile Error**: 87.14 pixels

---

### 2. YOLOv8 Object Detection Model Metrics
YOLOv8 was trained on the dataset using bounding boxes of a fixed size ($0.05 \times 0.05$) centered on the extracted eye centers.

#### 📊 Final Epoch (Epoch 50) Validation Metrics
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Precision** | 0.00155 | Fraction of correct detections among all detections |
| **Recall** | 0.11905 | Fraction of true eyes successfully detected |
| **mAP50** | 0.00091 | Mean Average Precision at IoU threshold = 0.50 |
| **mAP50-95** | 0.00014 | Mean Average Precision across IoU thresholds 0.50 to 0.95 |
| **Val Box Loss** | 4.232 | Bounding box coordinates regression loss |
| **Val Class Loss**| 4.177 | Classification loss (confidence loss) |
| **Val DFL Loss**  | 1.128 | Distribution Focal Loss (bounding box boundary regression) |

> [!IMPORTANT]
> **Performance Discrepancy Analysis**:
> YOLOv8 performed significantly worse than the custom ResNet50 regression head. This is because YOLOv8 is designed for general-purpose object detection and relies on **Intersection over Union (IoU)** overlap to update bounding boxes. Since the target eye bounding boxes are extremely small ($10 \times 10$ pixels), a tiny deviation of the predicted box center leads to zero IoU overlap ($IoU = 0$). This zero-overlap problem causes a vanishing gradient effect for box regressors and poor confidence scores. 
> 
> The ResNet50 regression head, in contrast, directly optimizes coordinates using the **Huber Loss**, which provides a continuous, linear gradient signal regardless of how far the prediction is from the target.

---

## 🛠️ How to Setup and Run

### Prerequisites
- Python 3.8 to 3.10
- CUDA-compatible GPU (recommended for training, though the code is runnable on CPU)

### Step 1: Install Dependencies
Clone the repository, set up a virtual environment, and install the package requirements:
```bash
# Set up virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Running the ResNet50 Regression Pipeline
1. Ensure the raw dataset `cyclone_dataset.h5` is in the project root directory.
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook cyclone_eye_detection.ipynb
   ```
3. Execute the notebook cells sequentially:
   - **Cells 2–7**: Examine dataset structure and load matrices.
   - **Cell 8**: Preprocess imagery (selects channels `0,1,3` and handles NaNs) and extracts eye centers from the mask channel.
   - **Cells 9–14**: Normalize labels to $[0, 1]$, visualize target labels, and partition data into train/validation sets.
   - **Cells 18–21**: Compile model using a custom ResNet50 regression head with **Huber Loss** and execute the training loop.
   - **Cells 22–36**: Evaluate results, plot metrics, run sample predictions, and save validation parameters and metrics files.

### Step 3: Running YOLOv8 Training
If you want to train the YOLOv8 model again or experiment with its parameters:
1. Running **Cell 32** in the notebook exports the dataset images and labels into the YOLO format inside the `datasets/cyclone_eye` directory.
2. You can train the YOLOv8 model directly from the command line using:
   ```bash
   yolo task=detect mode=train model=yolov8n.pt data=cyclone_coco.yaml epochs=50 imgsz=224 device=cpu
   ```
   *(Change `device=cpu` to `device=0` if you have an active GPU).*
3. The results will be saved in `runs/train/cyclone_eye_yolov8/`.

---

## ⚙️ Model Details & Design Rationale

For a deep dive into the dataset specifications, preprocessing steps, architectural designs, comparison of loss functions, and a comprehensive justification of why the regression model outperforms object detectors for this task, see the accompanying [PROJECT_DETAILS.md](PROJECT_DETAILS.md) file.
