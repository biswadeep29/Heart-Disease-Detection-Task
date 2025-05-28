
# 🫀 Heart Disease Classification Using Deep Learning on ECG Signals

## 📚 Description
This project involves classifying ECG signals as **Normal** or **Arrhythmia** using deep learning techniques. The signals are transformed into **scalogram images** using Continuous Wavelet Transform (CWT), and a **VGG16 transfer learning model** is trained for classification.

---

## 📁 Dataset

- **Source:** 
  - [MIT-BIH Arrhythmia Database](https://www.physionet.org/content/mitdb/1.0.0/)
  - [MIT-BIH Normal Sinus Rhythm Database](https://physionet.org/content/nsrdb/1.0.0/)

- **Classes:**
  - `Normal` ECG
  - `Arrhythmia` ECG

- **Preprocessing:**
  - Signals read using `wfdb` from `.dat` and `.hea` files
  - First 2 seconds of signal used from Lead I
  - CWT applied to generate scalogram images (using Morlet wavelet)
  - Images saved into class folders: `cwt_images/normal` and `cwt_images/arrhythmia`

---

## 🧠 Model Architecture

- **Model Type:** Transfer Learning using VGG16
- **Input Size:** 128 × 128 × 3
- **Base Model:** VGG16 with ImageNet weights (convolutional layers frozen)
- **Custom Classifier Head:**
  - `Flatten`
  - `Dense(64, activation='relu')`
  - `Dense(1, activation='sigmoid')`

- **Training:**
  - Optimizer: `Adam`
  - Loss: `BinaryCrossentropy`
  - Metrics: `Accuracy`
  - Epochs: `3`
  - Data split: `80%` training / `20%` validation

---

## 📈 Evaluation

- **Accuracy on Test Set:** 100%
- **Loss on Test Set:** 100%
- **Visualization:** Accuracy and loss curves plotted per epoch

---

## 🧪 Prediction Pipeline

User can input their own `.dat` and `.hea` ECG files.

### Steps:
- Load ECG signal from user-uploaded files
- Convert to scalogram using CWT
- Feed image to trained model
- Output prediction: `"Normal"` or `"Arrhythmia"`

```python
# Sample usage
signal, fs = load_user_signal('./user/', '1')
generate_scalogram(signal, 'user_scalogram.png')
result = predict_heart_condition('vgg_transfer_model.h5', 'user_scalogram.png')
print("Prediction:", result)

# Another Way
complete_pipeline('./user','16786','user_scalogram.png','ecg_model.h5',true_label='Normal')

