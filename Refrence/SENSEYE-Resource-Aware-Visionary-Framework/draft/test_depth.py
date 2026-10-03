import pyrealsense2 as rs

# Create a pipeline
pipeline = rs.pipeline()

# Configure the pipeline to stream both color and depth
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

# Wait for the first set of frames
frames = pipeline.wait_for_frames()

# Get the color and depth frames
color_frame = frames.get_color_frame()
depth_frame = frames.get_depth_frame()

if color_frame and depth_frame:
    print("Color and Depth frames received successfully.")
else:
    print("Failed to get frames.")

# Stop the pipeline after testing
pipeline.stop()
