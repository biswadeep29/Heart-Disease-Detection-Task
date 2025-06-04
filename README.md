

# Heart Disease Classification Using Deep Learning on ECG Signals

## Description
This project involves classifying ECG signals as **Arrhythmia**, **Atrial Fibrillation**, **Malignant Ventricular Ectopy**, **ST-Change** and **Normal** using deep learning techniques. The signals are transformed into scalogram images using Continuous Wavelet Transform (CWT), and a few CNN models are trained for multi-class classification of cardiac conditions.

---

## Dataset

- **Source:** 
  - [MIT-BIH Arrhythmia Database](https://www.physionet.org/content/mitdb/1.0.0/)
  - [MIT-BIH Atrial Fibrillation Database](https://physionet.org/content/afdb/1.0.0/)
  - [MIT-BIH Malignant Ventricular Ectopy Database](https://physionet.org/content/vfdb/1.0.0/)
  - [MIT-BIH ST Change Database](https://physionet.org/content/stdb/1.0.0/)
  - [MIT-BIH Normal Sinus Rhythm Database](https://physionet.org/content/nsrdb/1.0.0/)


- **Preprocessing:**
  - Signals read using `wfdb` from `.dat` and `.hea` files
  - First 2 seconds of signal used from Lead I
  - CWT applied to generate scalogram images (using Morlet wavelet)
  - Images saved into class folders: `cwt_images/normal`, `cwt_images/arrhythmia` etc...

---

## Model Used
- 2 Custom CNN Models
- VGG16
- ResNet50
- ResNet101
- Xception
- DenseNet121


## Training:
  - Optimizer: `Adam`
  - Loss: `Categorical Crossentropy`
  - Metrics: `Accuracy`
  - Epochs: `10`
  - Data split: `80%` training / `20%` validation

---

## Prediction Pipeline

User can input their own `.dat` and `.hea` ECG files.

### Steps:
- Load ECG signal from user-uploaded files
- Convert to scalogram using CWT
- Feed image to trained model
- Output prediction
