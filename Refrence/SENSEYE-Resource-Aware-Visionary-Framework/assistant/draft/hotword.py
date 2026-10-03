import sounddevice as sd
import numpy as np
import tempfile
import wave
import whisper
import time
import torch

# Load Whisper model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model = whisper.load_model("base").to(device)

# Configuration
HOTWORD = "hello"
SAMPLE_RATE = 16000
CHUNK_DURATION = 2  # Duration of each audio chunk (in seconds)

def record_audio(duration, fs):
    """Record audio for a specified duration."""
    print("Recording audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait for recording to finish
    return audio

def transcribe_audio(audio_data, fs):
    """Transcribe audio using Whisper."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        with wave.open(tmpfile.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(fs)
            wf.writeframes(audio_data.tobytes())
        # Transcribe audio
        result = whisper_model.transcribe(tmpfile.name, language="en")
    return result.get("text", "").strip()

def hotword_detection():
    """Continuously listen for the hotword."""
    print("Listening for the hotword...")
    while True:
        try:
            # Record a chunk of audio
            audio_data = record_audio(CHUNK_DURATION, SAMPLE_RATE)
            
            # Transcribe the audio chunk
            transcription = transcribe_audio(audio_data, SAMPLE_RATE)
            print(f"Transcription: {transcription}")

            # Check if the hotword is in the transcription
            if HOTWORD in transcription.lower():
                print("Hotword detected!")
                continue
        
                # break  # Exit after detecting the hotword

        except Exception as e:
            print(f"Error during hotword detection: {e}")

if __name__ == "__main__":
    hotword_detection()
