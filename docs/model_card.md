# Model Card: Adaptive Edge-AI Navigation System

## 1. Overview
The Adaptive Edge-AI Navigation architecture employs pretrained, open-source deep learning foundation models integrated with real-time algorithmic kinematic, temporal, and spatial reasoning modules. None of the underlying vision models are claimed as custom-trained from scratch; they represent rigorously evaluated off-the-shelf foundation weights adapted and accelerated for real-time edge execution.

---

## 2. Model 1: Real-Time Object Detector (YOLOv8n / YOLO11n)
- **Model Name**: YOLOv8n / YOLO11n (Nano Object Detection Network)
- **Model Type**: Single-stage convolutional neural network with anchor-free detection head and C2f feature aggregators
- **Version**: Ultralytics YOLOv8 (v8.1.0 engine, `yolo11n.pt` checkpoint)
- **Source**: Ultralytics GitHub Repository (`https://github.com/ultralytics/ultralytics`)
- **Input**: 3-channel RGB image ($640 	imes 480 	imes 3$), normalized $[0, 1]$, tensor precision FP16/FP32
- **Output**: Bounding box coordinates $[x_1, y_1, x_2, y_2]$, confidence score $[0.0, 1.0]$, and categorical class ID across 80 COCO categories
- **Purpose**: Real-time localization and semantic classification of potential physical obstacles (pedestrians, vehicles, bicycles, furniture) within the user's forward field of view
- **Limitations**:
  - Precision degrades in extreme low-light environments ($< 25	ext{ lux}$) due to loss of edge contrast
  - High-gloss floor reflections can occasionally trigger transient false positives ($conf \sim 0.28$)
  - Highly occluded objects ($> 50\%$ obscured) exhibit lower detection confidence
- **License**: GNU Affero General Public License v3.0 (AGPL-3.0) / Enterprise Commercial License

---

## 3. Model 2: Multi-Object Tracking & Association (BoT-SORT)
- **Model Name**: BoT-SORT (Booster Track Multi-Object Tracker)
- **Model Type**: Hybrid spatial-appearance tracking pipeline combining Kalman filtering, camera motion compensation, and visual re-identification embeddings
- **Version**: BoT-SORT implementation via Ultralytics / ByteTrack lineage
- **Source**: Aharon et al., *BoT-SORT: Robust Associations Multi-Pedestrian Tracking*, arXiv:2206.14651 (2022)
- **Input**: Current frame bounding boxes, confidence scores, and visual image crops
- **Output**: Persistent integer Track IDs, smoothed trajectory history, and state velocity vectors
- **Purpose**: Maintain obstacle identity across sequential video frames, enabling trajectory estimation and range-rate derivative calculations
- **Limitations**:
  - Abrupt camera angular velocities ($> 40^\circ/	ext{s}$) during brisk head/torso pivots can exceed Kalman association gates, causing transient ID switches
  - Long occlusions exceeding $25	ext{ frames}$ trigger track expiration and re-initialization
- **License**: MIT License

---

## 4. Model 3: Monocular Depth Foundation Model (Depth Anything V2)
- **Model Name**: Depth Anything V2 (`vits` Small Variant)
- **Model Type**: Vision Transformer (ViT-Small) monocular relative and metric depth estimation network
- **Version**: Depth Anything V2 Small (`depth_anything_v2_vits.pth`, 24.8M parameters)
- **Source**: Yang et al., *Depth Anything V2: A Foundation Model for Monocular Depth Estimation*, arXiv:2406.09414 (2024)
- **Input**: Preprocessed RGB tensor resized to $518 	imes 518$, normalized with ImageNet mean and standard deviation
- **Output**: Dense single-channel depth disparity map normalized to metric distance equivalents (meters)
- **Purpose**: Provide spatial range awareness without requiring bulky or power-hungry stereo rigs or physical LiDAR hardware
- **Limitations**:
  - Monocular depth estimation is subject to vertical baseline priors; partial vertical occlusion behind pillars causes transient depth overestimation ($+0.42	ext{m}$)
  - Untextured featureless white walls provide low disparity gradients
  - Native full-resolution inference on CPU requires $\sim 4.2	ext{s}$; mitigated by FP16 GPU acceleration and 2:1 cadence interleaving ($7.82	ext{ms}$ per frame)
- **License**: Apache License 2.0

---

## 5. Model 4: Speech Synthesizer / Audio Feedback Engine
- **Model Name**: Microsoft SAPI5 Voice Synthesizer / eSpeak Multi-Platform Engine
- **Model Type**: Offline parametric/formant voice synthesis driver via `pyttsx3`
- **Version**: `pyttsx3` v2.90
- **Source**: Python Package Index (`https://pypi.org/project/pyttsx3/`)
- **Input**: Short, formatted advisory string (average $7.8	ext{ words}$)
- **Output**: Low-latency mono/stereo acoustic waveform routed to system speaker or Bluetooth 5.2 wearable earbud
- **Purpose**: Deliver clear, non-visual spoken navigational guidance to visually impaired operators
- **Limitations**:
  - Fixed cadence and tone; lack of affective prosody modulation under varying stress levels
  - Audio output thread requires non-blocking priority buffering to prevent main perception loop blocking
- **License**: MPL-2.0 / LGPL-2.1
