#!/usr/bin/env python3
import logging
import time
import termcolor

from nano_llm import Agent, NanoLLM, ChatHistory
from nano_llm.utils import ArgParser, load_prompts
from nano_llm.plugins import VideoSource, VideoOutput
from jetson_utils import cudaMemcpy, cudaToNumpy

class VideoStream(Agent):
    """
    Captures video from `--video-input` and optionally streams processed frames to `--video-output`.
    When prompted, it analyzes the scene using an LLM.
    
    Example usage:
    ```bash
    python3 script.py --video-input /dev/video0 --video-output webrtc://@:8554/output
    ```
    """

    def __init__(self, video_input=None, video_output=None, **kwargs):
        """
        Args:
          video_input (str): The input video stream (e.g., RTSP, WebRTC, CSI, file path).
          video_output (str): The output stream (e.g., WebRTC, RTP, file path).
        """
        super().__init__()

        self.video_source = VideoSource(video_input, **kwargs)
        self.video_output = VideoOutput(video_output, **kwargs) if video_output else None

        self.video_source.add(self.on_video, threaded=False)

        if self.video_output:
            self.video_source.add(self.video_output)

        self.pipeline = [self.video_source]

        # Load LLM model
        self.model = NanoLLM.from_pretrained(
            kwargs.get("model", "Efficient-Large-Model/VILA1.5-3b"), 
            api=kwargs.get("api"),
            quantization=kwargs.get("quantization"), 
            max_context_len=kwargs.get("max_context_len"),
            vision_model=kwargs.get("vision_model"),
            vision_scaling=kwargs.get("vision_scaling"), 
        )

        assert self.model.has_vision

        # Chat history and prompts
        self.chat_history = ChatHistory(self.model, kwargs.get("chat_template"), kwargs.get("system_prompt"))
        self.prompts = ["Summarize the scene in one sentence.", "What is happening in this image?"]

    def on_video(self, image):
        """ Process a frame and optionally send it to output """
        logging.debug(f"Captured {image.width}x{image.height} frame from {self.video_source.resource}")

        # # Display frame in terminal (optional)
        # print("\nPress 'Enter' to analyze the current frame...")

        # user_input = input()
        # if user_input.lower() in ["q", "quit", "exit"]:
        #     logging.info("Exiting...")
        #     self.stop()
        #     return

        # Run LLM-based scene description
        self.chat_history.append("user", image=image)
        time_begin = time.perf_counter()

        self.chat_history.append("user", self.prompts[0], use_cache=True)
        embedding, _ = self.chat_history.embed_chat()
        
        print('\n🔍 Analyzing scene...')
        
        reply = self.model.generate(
            embedding,
            kv_cache=self.chat_history.kv_cache,
            max_new_tokens=50,  # Limit response length
            min_new_tokens=10,
            do_sample=True,
            repetition_penalty=1.2,
            temperature=0.7,
            top_p=0.9,
        )

        # Extract text response
        concise_reply = " ".join([token for token in reply][:20])  # Extract tokens correctly
        termcolor.cprint(f"📌 Scene Summary: {concise_reply}", "blue", flush=True)

        time_elapsed = time.perf_counter() - time_begin
        print(f"\n⏱️ Processing time: {time_elapsed*1000:.2f} ms | Rate: {1.0/time_elapsed:.2f} FPS")

        self.chat_history.reset()

        exit(0)

        # Send processed frame to output if enabled
        if self.video_output:
            self.video_output.output(image)

if __name__ == "__main__":
    parser = ArgParser(extras=["video_input", "video_output", "log", "model"])
    args = parser.parse_args()
    
    agent = VideoStream(**vars(args)).run()
