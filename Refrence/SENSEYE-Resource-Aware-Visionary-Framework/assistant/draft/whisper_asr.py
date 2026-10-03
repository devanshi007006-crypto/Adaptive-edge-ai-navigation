import whisper
import sounddevice as sd
import numpy as np
import tempfile
import wave

# Load Whisper model
whisper_model = whisper.load_model("base")

def record_audio(duration=3, fs=16000):
    """Record short audio."""
    print("Recording audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return audio

def transcribe_audio(audio_data, fs=16000):
    """Transcribe audio using Whisper."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        with wave.open(tmpfile.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(audio_data.tobytes())
        result = whisper_model.transcribe(tmpfile.name, language="en")
    return result.get("text", "").strip()

def main():
    hotword = "hello" # Change this to your hotword
    print("Listening for hotword...")
    
    while True:
        audio_data = record_audio()
        transcription = transcribe_audio(audio_data)
        print(f"Detected text: {transcription}")
        
        if hotword in transcription.lower():
            print("Hotword detected!")
            # Add logic for triggering your assistant
            break  # For demo purposes, exit after detecting the hotword

if __name__ == "__main__":
    main()
