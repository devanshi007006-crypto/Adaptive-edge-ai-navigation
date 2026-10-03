jetson-containers run $(autotag nano_llm) \
  python3 -m nano_llm.agents.video_query --api=mlc \
    --vision-api=hf \
    --model Efficient-Large-Model/VILA1.5-3b \
    --max-context-len 256 \
    --max-new-tokens 32 \
    --video-input /dev/video4 \
    --video-output webrtc://@:8556/vlm
    
# Efficient-Large-Model/VILA1.5-3b     		YYY
# Efficient-Large-Model/VILA-2.7b 		YY
# Efficient-Large-Model/VILA-7b 		Y

# liuhaotian/llava-v1.6-vicuna-7b		XXX
# NousResearch/Obsidian-3B-V0.5			X
