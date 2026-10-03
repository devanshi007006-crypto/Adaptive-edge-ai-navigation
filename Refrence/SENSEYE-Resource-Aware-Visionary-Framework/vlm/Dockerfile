# Use the existing image as base
FROM dustynv/nano_llm:r36.3.0

# Install paho-mqtt and other dependencies
RUN pip3 install paho-mqtt --index-url https://pypi.org/simple

# Set working directory for your project
WORKDIR /vlm

# Default command to run your script (you can adjust this if needed)
CMD ["python3", "/vlm/vision.py", "--api=mlc", "--model", "Efficient-Large-Model/VILA1.5-3b", "--max-context-len", "128", "--max-new-tokens", "32"]


## build the docker image
# docker build -t custom_nano_llm .
