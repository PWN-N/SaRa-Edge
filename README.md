# SaRa-Edge 🎙️

### Lightweight Edge AI Keyword Spotting for Low-Power IoT Devices

SaRa-Edge is an low latency and voice activator for edge devices based voice interaction system designed to reduce unnecessary cloud audio transmission.

Instead of continuously sending microphone audio to a cloud speech-recognition service, the system first performs **local keyword spotting** using a lightweight machine-learning model. Only after the custom keyword is detected will the system stream the following voice command to a server for full speech recognition.

---

## 🚀 Project Overview

Traditional voice-controlled IoT systems often send continuous audio to a cloud server for speech recognition.

This can cause:

- Higher network usage
- Increased latency
- Unnecessary cloud processing
- Privacy concerns
- Dependence on continuous internet connectivity

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
       No Keyword    Keyword Found
          │             │
          ▼             ▼
     Keep Listening   Start Streaming
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
