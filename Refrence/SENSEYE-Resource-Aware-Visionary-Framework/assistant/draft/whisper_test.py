import time
import whisper

# Load the model
model = whisper.load_model("tiny")

# Load or record an audio sample
audio_path = "response.wav"

# Measure inference time
start_time = time.time()
result = model.transcribe(audio_path, fp16=False)
end_time = time.time()

print("Transcription:", result['text'])
print(f"Inference Time: {end_time - start_time:.2f} seconds")
