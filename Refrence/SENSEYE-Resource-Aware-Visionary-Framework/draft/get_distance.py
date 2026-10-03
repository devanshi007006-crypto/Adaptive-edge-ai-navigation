import pyrealsense2 as rs
import numpy as np

# Configure depth stream
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # Get distance at the center of the frame
        width, height = depth_frame.get_width(), depth_frame.get_height()
        distance = depth_frame.get_distance(width // 2, height // 2)

        # Print distance
        print(f"Distance to center: {distance:.2f} meters")

finally:
    # Stop streaming
    pipeline.stop()
