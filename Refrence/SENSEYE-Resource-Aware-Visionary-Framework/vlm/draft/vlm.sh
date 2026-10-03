#!/bin/bash

jetson-containers run \
  -v /home/jetson/bme/vlm:/vlm \
  $(autotag nano_llm) \
  python3 /vlm/vlm.py \
  --model Efficient-Large-Model/VILA1.5-3b \
  --video-input rtsp://127.0.0.1:8554/live \
  --video-input-codec h264 \
  --video-output webrtc://@:8554/output