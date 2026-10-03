#!/usr/bin/env python3
#
# This multimodal example is a simplified version of the 'Live Llava' demo,
# wherein the same prompt (or set of prompts) is applied to a stream of images.
#
# You can run it like this (these options will replicate the defaults)
#
#    python3 -m nano_llm.vision.example \
#      --model Efficient-Large-Model/VILA1.5-3b \
#      --video-input "/data/images/*.jpg" \
#      --prompt "Describe the image." \
#      --prompt "Are there people in the image?"
#
# You can specify multiple prompts (or a text file) to be applied to each image,
# and the video inputs can be sequences of files, camera devices, or network streams.
#
# For example, `--video-input /dev/video0` will capture from a V4L2 webcam. See here:
# https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md
#
import time
import termcolor
import paho.mqtt.client as mqtt


from nano_llm import NanoLLM, ChatHistory
from nano_llm.utils import ArgParser, load_prompts
from nano_llm.plugins import VideoSource

from jetson_utils import cudaMemcpy, cudaToNumpy

# MQTT Configuration
MQTT_BROKER = "localhost"  # Change this if your broker is on another machine
MQTT_PORT = 1883
MQTT_TOPIC_SUB = "query/vlm"
MQTT_TOPIC_PUB = "response/vlm"

# parse args and set some defaults
args = ArgParser(extras=ArgParser.Defaults + ['prompt', 'video_input']).parse_args()
prompts = load_prompts(args.prompt)

if not prompts:
    prompts = ["Describe the image concisely."]
    
if not args.model:
    args.model = "Efficient-Large-Model/VILA1.5-3b"

if not args.video_input:
    #args.video_input = "/data/images/lake.jpg"
    #args.video_input = "/dev/video4"
    #args.video_input = "rtsp://127.0.0.1:8554/live"
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

# open the video stream
video_source = VideoSource(**vars(args), cuda_stream=0, return_copy=False)

# MQTT Callback Functions
def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT Broker with result code " + str(rc))
    client.subscribe(MQTT_TOPIC_SUB)

def on_message(client, userdata, msg):
    if msg.topic == MQTT_TOPIC_SUB and msg.payload.decode() == "vlm":
        print("Received MQTT Message: Processing VLM Task...")
        response_text = process_vlm_task()
        client.publish(MQTT_TOPIC_PUB, response_text)
        print("Response Sent to MQTT:", response_text)

def process_vlm_task():
    """Captures an image, runs the model, and returns the response."""
    img = video_source.capture()
    if img is None:
        return "Error: No image captured"

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

        response_text = "".join(reply.tokens)
        output_responses.append(response_text)

        for token in reply:
            termcolor.cprint(token, 'blue', end='\n\n' if reply.eos else '', flush=True)

        chat_history.append('bot', reply)

    time_elapsed = time.perf_counter() - time_begin
    print(f"time: {time_elapsed*1000:.2f} ms  rate: {1.0/time_elapsed:.2f} FPS")

    chat_history.reset()

    return "\n".join(output_responses)  # Return response text for MQTT publishing

# Initialize MQTT Client
mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

print("Waiting for MQTT messages on", MQTT_TOPIC_SUB)

# Keep script running indefinitely
while True:
    time.sleep(1)
