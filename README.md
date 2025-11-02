# ✍️ Handwriting to Text Recognition (CRNN Model)

## 📌 Overview

This project implements an end-to-end **Handwritten Text Recognition (HTR)** system using a **Convolutional Recurrent Neural Network (CRNN)** architecture.

The system converts handwritten word images into corresponding digital text. It effectively combines:
1.  **CNN layers** for visual feature extraction from the handwriting.
2.  **RNN layers (Bi-LSTMs)** for sequential context modeling.
3.  **CTC loss** for robust, alignment-free sequence decoding.

## ⚙️ System Workflow

### Dataset Preparation
* The **IAM Handwriting Dataset** is used for training and evaluation.
* Images are organized and labeled based on ground-truth transcriptions.

### Data Preprocessing
* Each image is converted to grayscale.
* Resized uniformly to **$128 \times 512$** for consistent input.
* Normalized and converted into PyTorch tensors.

### 🧠 Model Architecture

The CRNN pipeline sequentially processes the image to extract and decode features:

| Module | Purpose |
| :--- | :--- |
| **Feature Extraction (CNN)** | Convolutional layers extract spatial features, stroke patterns, and writing flow. |
| **Sequence Modeling (RNN)** | Bidirectional LSTMs capture sequential dependencies and context between characters. |
| **CTC Decoding** | **Connectionist Temporal Classification** maps the sequence output to variable-length character sequences, handling alignment dependencies. |

Input Image (1×128×512) │ [CNN Layers] │ Feature Maps │ [BiLSTM Layers] │ Sequence Output │ [CTC Loss] │ Predicted Text Output


### Evaluation
The model performance is measured using standard metrics: **Character Error Rate (CER)** and **Word Error Rate (WER)**.

---

## 📊 Results

The model was evaluated on unseen handwriting samples.

| Metric | Value |
| :--- | :--- |
| Samples Tested | **619** |
| Character Accuracy | **91.64%** |
| Character Error Rate (CER) | **0.0449** |
| Word Error Rate (WER) | **0.1470** |

### 📈 Performance Visualization

```python
import matplotlib.pyplot as plt

metrics = ['Character Accuracy', 'CER', 'WER']
values = [91.64, 0.0449, 0.1470]

plt.figure(figsize=(8, 5))
plt.plot(metrics, values, marker='o', color='teal', linewidth=2)
plt.title('Model Evaluation Metrics')
plt.ylabel('Value')
plt.grid(True)
plt.show()
```
🚀 How to Run
1️⃣ Clone the Repository
```
git clone [https://github.com/](https://github.com/)<your-username>/handwriting_to_text.git
cd handwriting_to_text
```

2️⃣ Install Dependencies
```
pip install -r requirements.txt
```
3️⃣ Prepare Dataset
Place the IAM Handwriting Dataset files inside the ./data/ directory or update the dataset path in dataloader.py.

4️⃣ Train the Model
```
python train.py
```
5️⃣ Evaluate the Model
```
python evaluate.py
```
6️⃣ Predict on Custom Image
```
python predict.py --image <path_to_image>
```

📂 Project Structure
```
handwriting_to_text/
│
├── data/                       # Dataset folder
├── models/                     # Trained models / checkpoints
├── outputs/                    # Predictions and result images
├── train.py                    # Model training script
├── evaluate.py                 # Model evaluation script
├── predict.py                  # Custom image prediction
├── dataloader.py               # Dataset loader and transformations
├── CRNN_model.py               # CRNN architecture definition
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```
🧰 Technologies Used
Python

PyTorch (for deep learning framework)

OpenCV

PIL (Pillow)

NumPy / Matplotlib

IAM Handwriting Dataset

📖 Future Enhancements
Integrate Transformer-based recognition models (e.g., Vision Transformers).

Add multi-language handwriting support.

Develop a web interface using Flask/Streamlit for easy demo.

Enable real-time handwriting recognition via webcam input.

👨‍💻 Author
Dilip Kannan
