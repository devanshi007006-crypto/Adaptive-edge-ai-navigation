import whisper, requests, os, sounddevice as sd, numpy as np, tempfile, wave
import faiss
from sentence_transformers import SentenceTransformer
import torch
import subprocess


# Optimization: Use a more efficient embedding model for Jetson Orin Nano
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Optimization: Explicitly use CUDA if available, with fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model = whisper.load_model("base").to(device)

# Configuration for local LLM server
llama_url = "http://127.0.0.1:8080/completion"

# Initial prompt to guide the LLaMA model's behavior
initial_prompt = ("You're an AI assistant specialized in AI development, embedded systems like the Jetson Nano, and Google technologies. "
                  "Answer questions clearly and concisely in a friendly, professional tone. Do not use asterisks and emojis, do not ask new questions "
                  "or act as the user. Keep replies short to speed up inference. If unsure, admit it and suggest looking into it further.")

# Documents to be used in Retrieval-Augmented Generation (RAG)
docs = [
    "I am you personal AI assistant",
    "Feel free to ask me anything about AI development",
    "I don't need internet connection.",
]

# Current directory and path for beep sound files (used to indicate recording start and end)
current_dir = os.path.dirname(os.path.abspath(__file__))
bip_sound = os.path.join(current_dir, "assets/bip.wav")
bip2_sound = os.path.join(current_dir, "assets/bip2.wav")

# Vector Database class to handle document embedding and search using FAISS
class VectorDatabase:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []
    
    def add_documents(self, docs):
        embeddings = embedding_model.encode(docs)  # Get embeddings for the docs
        self.index.add(np.array(embeddings, dtype=np.float32))  # Add them to the FAISS index
        self.documents.extend(docs)
    
    def search(self, query, top_k=3):
        query_embedding = embedding_model.encode([query])[0].astype(np.float32)
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        return [self.documents[i] for i in indices[0]]

# Create a VectorDatabase and add documents to it
db = VectorDatabase(dim=384)
db.add_documents(docs)

#  set audio output
def set_default_sink_if_available(desired_sink_name):
    """
    Set the default audio output device to the desired sink if it's available and not already selected.
    """
    try:
        # Get the current default sink
        result = subprocess.run(["pactl", "get-default-sink"], stdout=subprocess.PIPE, text=True, check=True)
        current_sink = result.stdout.strip()
        print(f"Current default sink: {current_sink}")

        # Check if the desired sink is already set
        if current_sink == desired_sink_name:
            print(f"{desired_sink_name} is already the default sink. Skipping.")
            return

        # List all available sinks
        result = subprocess.run(["pactl", "list", "short", "sinks"], stdout=subprocess.PIPE, text=True, check=True)
        available_sinks = [line.split("\t")[1] for line in result.stdout.splitlines()]
        print(f"Available sinks: {available_sinks}")

        # Check if the desired sink is available
        if desired_sink_name in available_sinks:
            # Set the desired sink as the default
            subprocess.run(["pactl", "set-default-sink", desired_sink_name], check=True)
            print(f"Default sink set to: {desired_sink_name}")
        else:
            print(f"{desired_sink_name} is not available. Skipping.")
    except subprocess.CalledProcessError as e:
        print(f"Error managing default sink: {e}")

def set_default_source_if_available(desired_source_name):
    """
    Set the default audio input device (microphone) to the desired source if it's available.
    """
    try:
        # Get the current default source
        result = subprocess.run(["pactl", "get-default-source"], stdout=subprocess.PIPE, text=True, check=True)
        current_source = result.stdout.strip()
        print(f"Current default source: {current_source}")

        # Check if the desired source is already set
        if current_source == desired_source_name:
            print(f"{desired_source_name} is already the default source. Skipping.")
            return

        # List all available sources
        result = subprocess.run(["pactl", "list", "short", "sources"], stdout=subprocess.PIPE, text=True, check=True)
        available_sources = [line.split("\t")[1] for line in result.stdout.splitlines()]
        print(f"Available sources: {available_sources}")

        # Check if the desired source is available
        if desired_source_name in available_sources:
            # Set the desired source as the default
            subprocess.run(["pactl", "set-default-source", desired_source_name], check=True)
            print(f"Default source set to: {desired_source_name}")
        else:
            print(f"{desired_source_name} is not available. Skipping.")
    except subprocess.CalledProcessError as e:
        print(f"Error managing default source: {e}")


# Play sound (beep) to signal recording start/stop
def play_sound(sound_file):
    os.system(f"aplay {sound_file}")

# Record audio using sounddevice
def record_audio(duration=1, fs=16000):
    """Record short audio for hotword detection or commands."""
    print("Recording audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return audio

# Transcribe recorded audio to text using Whisper
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

# Process user query using RAG and LLaMA
def rag_ask(query):
    """Generate a response using RAG."""
    context = " ".join(db.search(query))
    return ask_llama(query, context)

def ask_llama(query, context):
    """Send a query and context to the LLaMA server for completion."""
    data = {
        "prompt": f"{initial_prompt}\nContext: {context}\nQuestion: {query}\nAnswer:",
        "max_tokens": 80,
        "temperature": 0.7
    }
    response = requests.post(llama_url, json=data, headers={'Content-Type': 'application/json'})
    if response.status_code == 200:
        return response.json().get('content', '').strip()
    else:
        return f"Error: {response.status_code}"

# Text-to-speech using Piper TTS model
def text_to_speech(text):
    os.system(f'echo "{text}" | /home/jetson/piper/build/piper --model /usr/local/share/piper/models/en_US-lessac-medium.onnx --output_file response.wav && aplay response.wav')

# Main assistant logic after hotword detection
def assistant_logic():
    """Handles commands after the hotword is detected."""
    print("Hotword detected! Assistant is now active.")
    text_to_speech("I'm listening. How can I assist?")
    
    while True:
        play_sound(bip_sound)  # Start beep
        audio_data = record_audio(duration=5)  # Record user command
        play_sound(bip2_sound)  # End beep
        query = transcribe_audio(audio_data)
        print(f"User said: {query}")
        
        if not query:
            text_to_speech("Going to sleep mode.")
            print("No input detected. Returning to sleep mode.")
            return  # Exit to sleep mode
        
        if "turn off" in query.lower() or "exit" in query.lower():
            print("Exiting assistant. Goodbye!")
            text_to_speech("Exiting assistant. Goodbye!")
            exit(0)

        # Check if the user wants to run a specific script
        if "open camera" in query.lower():
            print("Running 'vlm' bash script...")
            text_to_speech("Running the VLM script now.")
            try:
                subprocess.run(["/home/jetson/bme/vlm/llava.sh"], check=True)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error running script: {e}")
                text_to_speech("There was an error running the script.")
            continue
        
        response = rag_ask(query)
        response = response.replace("*", "")  # Remove asterisks from the response
        print(f"Agent response: {response}")
        if response:
            text_to_speech(response)

# Main function with hotword detection
def main():

    # Set the default sink to Jabra
    sink_name = "alsa_output.usb-GN_Netcom_A_S_Jabra_EVOLVE_20_MS_A009E07823660A-00.analog-stereo"
    set_default_sink_if_available(sink_name)

     # Set the default source to Jabra
    source_name = "alsa_input.usb-GN_Netcom_A_S_Jabra_EVOLVE_20_MS_A009E07823660A-00.mono-fallback"
    set_default_source_if_available(source_name)

    """Continuously listens for the hotword and triggers assistant logic."""
    hotword = "hello"  # Set your hotword
    print("Listening for hotword...")
    
    while True:
        audio_data = record_audio(duration=3)  # Record short audio for hotword detection
        transcription = transcribe_audio(audio_data)
        print(f"Detected text: {transcription}")
        
        if hotword in transcription.lower():
            assistant_logic()

if __name__ == "__main__":
    main()
