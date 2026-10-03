from gtts import gTTS
import subprocess

# Generate the MP3 file
text = "Testing audio playback."
tts = gTTS(text, lang="en")
tts.save("test_audio.mp3")

# Play the MP3 file
subprocess.run(["ffplay", "-nodisp", "-autoexit", "test_audio.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
