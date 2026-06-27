# 📖 Project Details & Technical Rationale

This document provides a deep dive into the engineering, architecture, and design decisions behind the **Cyclone Eye Detection** project. It explains what methodologies were used, why they were chosen, and why alternative approaches were rejected.

---

## 🔍 1. Dataset & Preprocessing Details

### Dataset Source and Shape
The raw data is stored in `cyclone_dataset.h5` inside a dataset named `'matrix'` of shape `(4580, 201, 201, 4)`. This represent 4,580 satellite observations, each being a $201 \times 201$ grid with 4 multi-spectral channels.

### Channel Selection & Handling of NaNs
- **Channel 0 & 1**: High-resolution satellite observation channels.
- **Channel 2**: Contains invalid measurements represented by `NaN` values.
- **Channel 3**: Storm eye mask.

```python
# Select channels 0, 1, and 3, ignoring Channel 2 entirely
images = data[:,:,:, [0,1,3]]
```
> [!IMPORTANT]
> **Why we ignored Channel 2**: Deep neural networks cannot propagate gradients through `NaN` (Not-a-Number) values. A single `NaN` value in the input tensor will propagate through convolution operations, causing weights, activations, and losses to become `NaN` (commonly referred to as "gradient explosion" or "NaN propagation"). Instead of trying to impute or fill the channel, it was completely discarded.

### Label Extraction from Masks
The dataset does not contain predefined $(x, y)$ coordinates for the cyclone eye center. Instead, we extracted them programmatically from Channel 3 (mask):
1. **Clean NaNs**: Replaced any missing values in the mask with 0.0.
2. **Thresholding**: Found positive mask values and set a dynamic threshold at the 50th percentile of these positive values to isolate the peak eye intensity.
3. **Centroid Calculation**: Calculated the coordinate centroid, weighted by mask intensity:
$$x_c = \frac{\sum (x_i \cdot w_i)}{\sum w_i}, \quad y_c = \frac{\sum (y_i \cdot w_i)}{\sum w_i}$$
This approach yields sub-pixel float coordinates indicating the physical center of mass of the eye region.
4. **Coordinate Normalization**: Coordinates in $[0, 201]$ were normalized by dividing by 201.0 to fit the range $[0, 1]$. Normalizing coordinates prevents scale imbalances and allows the final Dense layer to use a `sigmoid` activation function to naturally bound the output range.

---

## 🔄 2. Data Augmentation Design Constraints

Data augmentation is used to prevent overfitting. However, a strict physics-based constraint was applied:
* **Allowed**: Rotation (up to 10°), width shift (10%), height shift (10%).
* **Disabled**: Horizontal and vertical flipping (`horizontal_flip=False`, `vertical_flip=False`).

> [!CAUTION]
> **Why Flipped Images Were Disallowed**:
> Cyclonic storms are meteorological phenomena governed by the **Coriolis Effect**. Due to the rotation of the Earth, cyclones rotate **counter-clockwise in the Northern Hemisphere** and **clockwise in the Southern Hemisphere**. 
> 
> If we flip a satellite image of a Northern Hemisphere cyclone horizontally, it appears to rotate clockwise (representing a Southern Hemisphere storm). However, storm structures, cloud bands, and wind fields possess subtle asymmetry related to their hemisphere. Fictitiously flipping images generates physically impossible combinations of atmospheric features that can confuse the model and degrade its generalization on real physical systems.

---

## 🏗️ 3. Model Architecture Selection

### Backbone: ResNet50 (Pretrained on ImageNet)
- **Feature Extraction**: Convolutional Neural Networks (CNNs) are highly effective at capturing spatial hierarchies. **ResNet50** was chosen as the backbone due to its residual connections (skip connections) which solve the vanishing gradient problem, allowing deep layers to learn fine structural details.
- **Pretrained Weights**: The model was initialized with weights trained on **ImageNet**. Although satellite imagery is different from natural images, the early layers of ResNet50 are excellent edge and texture detectors, which are critical for detecting cloud margins and storm spiral patterns.
- **Layer Freezing**: The first 145 layers of ResNet50 were frozen (locked during training), while the last 30 layers were unfrozen for fine-tuning. This preserves general feature extractors while tailoring the high-level representation layers to storm geometry.

### Custom Regression Head
The classification top of ResNet50 was replaced with a custom coordinate regression network:
1. **GlobalAveragePooling2D**: Flattens the $7 \times 7 \times 2048$ feature map to a $2048$-dimensional vector while keeping spatial invariant statistics.
2. **Dense (1024) + ReLU + Batch Normalization + Dropout (0.4)**: Learns complex non-linear combinations of features while preventing overfitting.
3. **Dense (512) + ReLU + Batch Normalization + Dropout (0.3)**: Continues feature reduction.
4. **Dense (256) + Dropout (0.2)**: Final fully connected layers.
5. **Dense (2) + Sigmoid**: Outputs the coordinate pair $(x, y)$. Sigmoid clamps outputs strictly within the range $[0, 1]$.

---

## 📉 4. Loss Function Design: Huber Loss vs. MSE vs. MAE

For bounding box regression, selecting the correct loss function is critical:

```python
tf.keras.losses.Huber(delta=0.1)
```

### Why Huber Loss was Chosen
1. **Mean Squared Error (MSE)** penalizes errors quadratically ($L = e^2$). In noisy satellite data, if a storm has a highly sheared or ambiguous eye, a model might predict a point far away from the centroid. MSE will penalize this heavily, causing the gradients to change drastically and disrupt the training process for cleaner images.
2. **Mean Absolute Error (MAE / L1)** penalizes errors linearly ($L = |e|$). This is robust to outliers, but the gradient is constant and non-differentiable at $e=0$. This causes coordinates to jump around and oscillate instead of smoothly converging near the exact center.
3. **Huber Loss** acts as a hybrid:
$$L_{\delta}(a) = \begin{cases} \frac{1}{2}a^2 & \text{for } |a| \le \delta \\ \delta(|a| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$
For small errors ($|a| \le \delta$), it is quadratic (MSE-like, ensuring smooth convergence and clean gradients). For large errors, it is linear (MAE-like, limiting the influence of extreme outlier storms). We set $\delta = 0.1$ to achieve the ideal balance.

---

## ⚖️ 5. Why Not Other Architectures?

### Why Regression Outperformed YOLOv8 (Object Detection)
Object detection models like **YOLOv8** identify bounding boxes using **Intersection over Union (IoU)** matching. During preprocessing for YOLOv8, we created tiny artificial bounding boxes ($0.05 \times 0.05$ of the image height/width) around the eye.

This led to poor results (Precision: 0.0015, Recall: 0.1190) due to:
* **The Zero-IoU Problem**: Because the target box is extremely small (around $10 \times 10$ pixels), a prediction that is slightly off will have an IoU of $0.0$. In YOLOv8's loss function, when $IoU=0$, the gradient for box coordinates becomes zero. The model receives no signal on which direction to move the prediction, causing training to stall.
* **Continuous Gradients in Regression**: In our custom ResNet50 model, the Huber loss measures the Euclidean distance between predicted and true coordinates. Even if the prediction is 100 pixels away from the true center, the loss function provides a strong, direct gradient pointing towards the center.

### Why Not U-Net Segmentation?
A semantic segmentation network (e.g., **U-Net**) could be trained to predict the eye mask, and we could calculate the centroid of the predicted mask during post-processing.
* **Why Rejected**:
  * **Memory Complexity**: Segmentation models generate a full mask of size $201 \times 201$ pixels, requiring large decoders, heavy memory footprints, and slow runtimes.
  * **Direct vs. Indirect Output**: A regression model maps features directly to coordinates $(x,y)$ using simple dense layers, resulting in faster inference speeds (perfect for real-time tracking) with fewer parameters.
  * **Label Dependency**: Segmentation models require pixel-perfect masks for training, which are highly subjective and difficult to annotate for weak cyclones, whereas center coordinates are easier to estimate.
