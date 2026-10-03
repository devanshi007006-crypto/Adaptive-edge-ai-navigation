cd
cd llama.cpp/build
./bin/llama-server -m ../models/gemma-2-2b-it-Q4_K_S.gguf -p 5000 -t 4 -c 1024 --gpu-layers 30
