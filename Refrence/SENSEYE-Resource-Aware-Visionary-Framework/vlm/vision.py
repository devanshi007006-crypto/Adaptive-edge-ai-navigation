#!/usr/bin/env python3
#
# This multimodal example is a simplified version of the 'Live Llava' demo,
# wherein the same prompt (or set of prompts) is applied to a stream of images.

import time
import termcolor
import os
from PIL import Image
import paho.mqtt.client as mqtt

from nano_llm import NanoLLM, ChatHistory
from nano_llm.utils import ArgParser, load_prompts

# MQTT Configuration
MQTT_BROKER = "localhost"  # Change this if your broker is on another machine
MQTT_PORT = 1883
MQTT_TOPIC_PUB = "response/vlm"  # Topic to publish VLM results

# parse args and set some defaults
args = ArgParser(extras=ArgParser.Defaults + ['prompt', 'video_input']).parse_args()
prompts = load_prompts(args.prompt)

if not prompts:
    prompts = ["Describe the image concisely."]
    
if not args.model:
    args.model = "Efficient-Large-Model/VILA1.5-3b"

if not args.video_input:
    # Default video input is set to '/vlm/frame.jpg'
    args.video_input = "/vlm/frame.jpg"
    
print(args)

# load vision/language model
model = NanoLLM.from_pretrained(
    args.model, 
    api=args.api,
    quantization=args.quantization, 
    max_context_len=args.max_context_len,
    vision_model=args.vision_model,
    vision_scaling=args.vision_scaling, 
)

assert(model.has_vision)

# create the chat history
chat_history = ChatHistory(model, args.chat_template, args.system_prompt)

# MQTT Callback Functions
def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT Broker with result code " + str(rc))

def process_vlm_task():
    """Captures an image, runs the model, and publishes the response."""
    frame_path = "/vlm/frame.jpg"
    
    if os.path.exists(frame_path):
        print("Frame found, processing...")
        # Load the image using PIL (Python Imaging Library)
        try:
            img = Image.open(frame_path)
            print(f"Image size: {img.size}, Mode: {img.mode}")  # Debugging image info
            img = img.convert("RGB")  # Ensure the image is in RGB format
        except Exception as e:
            print(f"Error loading image: {e}")
            return

        # Adding debugging to confirm image is passed correctly
        print(f"Image successfully loaded. Shape: {img.size}, Mode: {img.mode}")

        chat_history.append('user', image=img)
        time_begin = time.perf_counter()
        output_responses = []

        for prompt in prompts:
            chat_history.append('user', prompt, use_cache=True)
            embedding, _ = chat_history.embed_chat()

            print('>>', prompt)

            reply = model.generate(
                embedding,
                kv_cache=chat_history.kv_cache,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                do_sample=args.do_sample,
                repetition_penalty=args.repetition_penalty,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            print(f"Raw model response: {reply.tokens}")  # Debugging: print the raw tokens

            # Check if model generated any tokens
            if not reply.tokens:
                print("Model did not generate any tokens.")  # Debugging

            response_text = "".join(reply.tokens) if reply.tokens else "No response generated."
            print(f"Generated reply: {response_text}")  # Debugging model reply

            output_responses.append(response_text)

            # Instead of printing to terminal, collect tokens for the response
            collected_response = ""
            for token in reply:
                collected_response += token
                termcolor.cprint(token, 'blue', end='\n\n' if reply.eos else '', flush=True)

            chat_history.append('bot', reply)

        time_elapsed = time.perf_counter() - time_begin
        print(f"time: {time_elapsed*1000:.2f} ms  rate: {1.0/time_elapsed:.2f} FPS")

        chat_history.reset()

        # Delete the frame file after processing
        try:
            os.remove(frame_path)
            print("Frame deleted after processing")
        except Exception as e:
            print(f"Error deleting frame: {e}")

        response_text = collected_response if collected_response else "No response generated."
        
        # Remove unwanted tokens like "</s>" before publishing
        response_text = response_text.replace("</s>", "").strip()

        print(f"Generated response: {response_text}")

        # Publish the response directly without checking if it's valid
        try:
            print(f"Publishing to MQTT topic {MQTT_TOPIC_PUB}: {response_text}")
            mqtt_client.publish(MQTT_TOPIC_PUB, response_text)
            print("Published response to MQTT:", response_text)
        except Exception as e:
            print(f"Error publishing to MQTT: {e}")

        # Ensure MQTT message is processed
        mqtt_client.loop()  # This ensures any pending message gets processed

    else:
        print("No frame found.")

# Initialize MQTT Client
mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect

mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

print("Waiting for frame.jpg to process and publish results...")

# Keep script running indefinitely
while True:
    process_vlm_task()
    time.sleep(1)  # Wait for the next frame
