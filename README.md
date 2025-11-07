# AI-Powered Image Classification System: Technical Report

**Project Title:** Transfer Learning (MobileNetV2) for CIFAR-10 Classification
**Candidate Name:** Anjali Chebathina
**Date Submitted:** 06-11-2025

---

## 1. Objective and Implementation

The goal was to build an advanced AI model using **Transfer Learning** (MobileNetV2), deploy it locally via a **Streamlit web application**, and ensure the model could be **saved and loaded** for inference.

---

## 2. Model Architecture 

### 2.1 Model Choice: MobileNetV2
**MobileNetV2** was chosen as the base model. It is a powerful, lightweight architecture pre-trained on the massive **ImageNet** dataset. This allows us to leverage its learned features for our (relatively small) CIFAR-10 dataset, leading to faster training and higher accuracy.

### 2.2 Implementation
1.  **Load Base Model:** The MobileNetV2 model was loaded *without* its final classification layer (`include_top=False`).
2.  **Freeze Layers:** The weights of the entire base model were **frozen** (`base_model.trainable = False`). This prevents the pre-trained knowledge from being destroyed during training.
3.  **Add Custom Head:** A new classification "head" was added on top:
    * `GlobalAveragePooling2D`: To flatten the features.
    * `Dense(128, activation='relu')`: A hidden layer for custom learning.
    * `Dropout(0.5)`: To prevent the new head from overfitting.
    * `Dense(10, activation='softmax')`: The final output layer for our 10 CIFAR-10 classes.

---

## 3. Data Preprocessing

The data pipeline (`train_model.py`) was modified significantly for Transfer Learning:

1.  **Image Resizing:** All CIFAR-10 images (originally 32x32) were resized to **224x224** using OpenCV (`cv2`), the required input size for MobileNetV2.
2.  **Normalization:** Instead of just dividing by 255, we used the `preprocess_input` function specific to MobileNetV2, which normalizes images to the range `[-1, 1]`.
3.  **Data Split:** A standard 70% Training, 15% Validation, and 15% Test split was used.

---

## 4. Model Training & Optimization 
* **Optimizer:** `Adam` optimizer was used.
* **Optimization:** **Early Stopping** (`patience=10`) was implemented. This monitored the validation loss and stopped training automatically when the model ceased to improve, preventing wasted time.
* **Saving:** A `ModelCheckpoint` callback was used to **automatically save only the best version** of the model (based on validation loss) to `deliverables/model_transfer_learning.h5`.

---

## 5. UI and Deployment

* **Mini UI (Streamlit):** A web application was built using Streamlit (`app.py`).
* **Model Loading (Task 3):** The app **loads the saved `model_transfer_learning.h5`** file on startup.
* **Real-time Inference:** The UI allows a user to **upload a custom image**. The app preprocesses this image (resizes to 224x224, normalizes) and feeds it to the loaded model to get an instant prediction with a confidence score.
* **Local Deployment (Task 4):** The app is deployed locally by running the command `streamlit run app.py` in the terminal.

---

## 6. Final Test Results

<img width="1912" height="908" alt="Screenshot 2025-11-07 152931" src="https://github.com/user-attachments/assets/2f321dc5-4e0e-4482-9b0f-cdfd9144c303" />



### 6.1 Final Metrics

| Metric | Value |
| :--- | :--- |
| **Final Test Loss** | **0.7432** |
| **Final Test Accuracy** | **0.9348** |

---
### Contact
Mail: anjalichebathina@gmail.com
