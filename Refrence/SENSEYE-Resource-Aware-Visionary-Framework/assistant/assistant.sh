#!/bin/bash
cd /home/jetson/bme/assistant 

# Function to run llama-server in the background and monitor it
start_llama_server() {
    while true; do
        echo "Starting llama-server..."
        cd ~/bme/assistant/llama.cpp/build || exit 1
        ./bin/llama-server -m ../models/qwen2-0.5b-q4_k_m.gguf -p 5000 -t 4 -c 1024 --gpu-layers 30 
        # gemma-2-2b-it-Q4_K_S.gguf # qwen2-0.5b-q4_k_m.gguf
        
        echo "llama-server crashed or exited. Restarting in 5 seconds..."
        sleep 5
    done
}

# Function to run assistant.py in the foreground and monitor it

start_assistant() {
    while true; do
        echo "Starting assistant.py..."
        python3 /home/jetson/bme/assistant/assistant.py
        
        echo "assistant.py crashed or exited. Restarting in 5 seconds..."
        sleep 5
    done
}

# Start llama-server in the background
start_llama_server &
LLAMA_PID=$!

# Run assistant.py in the foreground
start_assistant

# When assistant.py exits, kill llama-server
echo "Stopping llama-server..."
kill "$LLAMA_PID"
echo "Done."
