jetson-containers run $(autotag nano_llm) \
  python3 -m nano_llm.vision.video \
    --vision-api=hf \
    --model Efficient-Large-Model/VILA1.5-3b \
    --max-images 8 \
    --max-new-tokens 16 \
    --video-input /dev/video4 \
    --video-output webrtc://@:8556/output \
    --prompt 'caption the video concisely'

# Efficient-Large-Model/VILA1.5-3b     		YYY
# Efficient-Large-Model/VILA-2.7b 		YY
# Efficient-Large-Model/VILA-7b 		Y

# liuhaotian/llava-v1.6-vicuna-7b		XXX
# NousResearch/Obsidian-3B-V0.5			XXX
