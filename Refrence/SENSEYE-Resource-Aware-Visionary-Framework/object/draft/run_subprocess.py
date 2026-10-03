import subprocess
import os

# Set PulseAudio environment variables for the correct user
os.environ["PULSE_SERVER"] = "/run/user/1000/pulse/native"
os.environ["XDG_RUNTIME_DIR"] = "/run/user/1000"

# Run the playback script as the correct user
try:
    subprocess.run(
        ["sudo", "-u", "jetson", "python3", "play_audio.py", "object_detected.mp3"],
        env=os.environ,
        check=True,
    )
    print("Audio played successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error while trying to play audio: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
