````markdown
# SaRa-Edge 🎙️

### Lightweight Edge AI Keyword Spotting for Low-Power IoT Devices

SaRa-Edge is a **low-latency, voice-activated edge AI system** designed for voice interaction on low-power IoT devices.

Instead of continuously sending microphone audio to a cloud speech-recognition service, SaRa-Edge performs **local keyword spotting (KWS)** on the edge device. Only when the custom keyword is detected does the system begin streaming the following voice command to a server for full speech recognition.

This approach helps reduce unnecessary network usage, latency, cloud processing, and continuous transmission of private audio.

---

## 🚀 Project Overview

Traditional voice-controlled IoT systems often continuously transmit microphone audio to a cloud server for speech recognition.

This can result in:

- 🌐 Higher network usage
- ⏱️ Increased latency
- ☁️ Unnecessary cloud processing
- 🔐 Privacy concerns
- 📶 Dependence on continuous internet connectivity

### SaRa-Edge Approach

```text
                 Microphone
                     │
                     ▼
              Audio Capture
                     │
                     ▼
             MFCC Feature Extraction
                     │
                     ▼
             Lightweight KWS Model
                     │
              ┌──────┴──────┐
              │             │
         No Keyword     Keyword Found
              │             │
              ▼             ▼
       Keep Listening    Start Streaming
                            │
                            ▼
                          Wi-Fi
                            │
                            ▼
                       ASR Server
                            │
                            ▼
                     Speech → Text
                            │
                            ▼
                   Command Processing
                            │
                            ▼
                       IoT Action
````

---

## 🎯 Key Features

* Edge-based keyword spotting
* Low-latency wake-word detection
* INT8 quantized neural network
* Lightweight DS-CNN architecture
* MFCC-based audio feature extraction
* I2S digital microphone support
* Designed for ESP32-S3
* Local processing before cloud communication
* Reduced unnecessary audio transmission
* Designed with a **≤256 KB runtime RAM target**

---

## 🧠 Machine Learning Model

SaRa-Edge uses a lightweight **Depthwise Separable Convolutional Neural Network (DS-CNN)** for keyword spotting.

### Model Input

| Parameter      | Value     |
| -------------- | --------- |
| Sample Rate    | 16 kHz    |
| Audio Duration | 2 seconds |
| Audio Samples  | 32,000    |
| MFCC Features  | 13        |
| Feature Shape  | 13 × 63   |

### Classes

| Label | Class   |
| ----: | ------- |
|     0 | Keyword |
|     1 | Unknown |
|     2 | Noise   |

---

## 📊 Model Performance

The trained floating-point DS-CNN achieved approximately:

**Test Accuracy: ~91%**

After INT8 quantization:

**INT8 Test Accuracy: 93.33%**

### INT8 Classification Results

| Class       | Precision |   Recall | F1-Score |
| ----------- | --------: | -------: | -------: |
| Keyword     |      0.87 |     1.00 |     0.93 |
| Unknown     |      0.95 |     0.90 |     0.92 |
| Noise       |      1.00 |     0.90 |     0.95 |
| **Overall** |  **0.94** | **0.93** | **0.93** |

### Confusion Matrix

```text
                 Predicted
              Keyword Unknown Noise

Actual Keyword    20      0      0
Actual Unknown     2     18      0
Actual Noise       1      1     18
```

---

## 📦 Model Size

The fully INT8 quantized TFLite model is approximately:

**19.14 KB**

The model is converted into a C/C++ header for embedded deployment:

```text
kws_model.h
```

---

## 🎤 Hardware

### Target Platform

* ESP32-S3
* INMP441 I2S MEMS microphone

### INMP441 Connection

Example GPIO configuration:

| INMP441 | ESP32-S3 |
| ------- | -------- |
| VDD     | 3.3V     |
| GND     | GND      |
| SCK     | GPIO 4   |
| WS      | GPIO 5   |
| SD      | GPIO 6   |
| L/R     | GND      |

> GPIO assignments may be changed depending on the exact ESP32-S3 development board being used.

---

## 🔊 Audio Processing Pipeline

The microphone captures audio at:

```text
16,000 Hz
```

The system collects:

```text
2 seconds = 32,000 audio samples
```

The audio is converted into MFCC features:

```text
Raw Audio
   ↓
16 kHz PCM
   ↓
MFCC
   ↓
13 × 63 Feature Matrix
   ↓
Normalization
   ↓
INT8 Quantization
   ↓
DS-CNN
```

---

## ⚡ Edge AI Deployment

The model is designed to run locally on the ESP32-S3 using **TensorFlow Lite Micro**.

The intended runtime pipeline is:

```text
INMP441
   │
   ▼
ESP32-S3
   │
   ▼
Audio Capture
   │
   ▼
MFCC
   │
   ▼
Normalization
   │
   ▼
INT8 DS-CNN
   │
   ├───────────────┐
   │               │
No Keyword    Keyword Detected
   │               │
   ▼               ▼
Continue       Start Wi-Fi
Listening          │
                   ▼
              ASR Server
```

---

## 💾 Memory Constraint

A major design requirement is:

```text
Runtime RAM ≤ 256 KB
```

The **19.14 KB model size should not be confused with runtime RAM usage**.

Actual embedded memory usage includes:

* TensorFlow Lite Micro tensor arena
* Audio buffers
* MFCC processing buffers
* Model tensors
* Runtime/application memory

The ESP32-S3 deployment will be used to measure the actual runtime memory requirement.

---

## 🛠️ Technology Stack

### Machine Learning

* Python
* TensorFlow / Keras
* NumPy
* Librosa
* TensorFlow Lite
* INT8 Quantization

### Embedded

* ESP32-S3
* ESP-IDF
* TensorFlow Lite Micro
* INMP441 I2S MEMS microphone

### Development

* VS Code
* Git
* GitHub

---

## 📁 Project Structure

```text
SaRa-Edge/
│
├── dataset/
│   ├── keyword/
│   ├── unknown/
│   └── noise/
│
├── features/
│   ├── X.npy
│   └── y.npy
│
├── models/
│   ├── kws_dscnn_v1.keras
│   ├── kws_dscnn_int8_v2.tflite
│   └── normalization_v2.npz
│
├── main/
│   ├── CMakeLists.txt
│   ├── main.cpp
│   └── kws_model.h
│
├── extract_features.py
├── train_DSCNN.py
├── test_int8.py
├── convert_to_tflite_int8_v2.py
├── convert_model.py
├── CMakeLists.txt
└── README.md
```

---

## 🔄 Development Workflow

### 1. Dataset Preparation

Audio samples are organized into:

```text
keyword/
unknown/
noise/
```

### 2. Feature Extraction

Audio is converted into MFCC features:

```text
Audio → MFCC → 13 × 63
```

### 3. Model Training

A lightweight DS-CNN is trained for the three classes.

### 4. INT8 Quantization

The trained model is converted to a fully INT8 TensorFlow Lite model.

### 5. Model Testing

The INT8 model is evaluated on the test dataset.

Current result:

```text
93.33% accuracy
```

### 6. Embedded Deployment

The `.tflite` model is converted into:

```text
kws_model.h
```

and integrated into the ESP32-S3 firmware.

### 7. Runtime Optimization

The tensor arena and application memory are measured to ensure the system meets:

```text
≤ 256 KB RAM
```

### 8. Voice Activation

Once the keyword is detected:

```text
Keyword detected
       ↓
Enable Wi-Fi / streaming
       ↓
Send following voice command
       ↓
ASR Server
       ↓
Speech-to-Text
       ↓
Command Processing
       ↓
IoT Action
```

---

## 🔮 Future Improvements

* Further reduce runtime RAM usage
* Optimize MFCC computation for ESP32-S3
* Integrate Wi-Fi communication
* Connect to an ASR server
* Add command processing
* Improve keyword robustness in noisy environments
* Evaluate end-to-end latency
* Measure CPU utilization
* Optimize low-power operation
* Improve wake-word detection under real-world conditions

---

## 🎯 Project Goal

SaRa-Edge aims to demonstrate that **keyword spotting can be performed efficiently at the edge**, allowing an IoT device to remain locally responsive while reducing unnecessary cloud audio transmission.

The goal is to build a system that is:

**Low-Latency + Lightweight + Privacy-Aware + Efficient + Edge-Based**

---

## 📜 License

This project is intended for educational and research purposes.

```
```
