# install dependencies:
```

pip install openai-whisper
pip install numpy==1.24.3
pip install sounddevice
pip install faiss-cpu
pip install gtts
pip install paho-mqtt
sudo apt update 
sudo apt install mosquitto mosquitto-clients
sudo apt install mpg123

```

# Build and Configure llama.cpp
```
cd
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build
cd build/
cmake .. -DGGML_CUDA=ON -DMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DGGML_CUDA_ARCHS="86"
make -j$(nproc)
```

# Download the Gemma model and mode to directory:
```
wget -O gemma-2-2b-it-Q4_K_S.gguf "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_S.gguf?download=true"

mv gemma-2-2b-it-Q4_K_S.gguf ../models/
```

# Install Torch 

jetson inference
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcusparselt0 libcusparselt-dev
pip3 install -r requirements.txt
```

# Optional

### benchmark 
```
./bin/llama-cli -m ../models/gemma-2-2b-it-Q4_K_S.gguf -p "Artificial intelligence is" -n 128 -ngl 999
```
### Launch Server
```
./bin/llama-server -m ../models/gemma-2-2b-it-Q4_K_S.gguf -p 5000 -t 4 -c 1024 --gpu-layers 30
```

### Run Local Server
./bin/llama-server -m ../models/gemma-2-2b-it-Q4_K_S.gguf -p 5000 -t 4 -c 1024 --gpu-layers 30 --host 192.168.x.x (replace with IP)