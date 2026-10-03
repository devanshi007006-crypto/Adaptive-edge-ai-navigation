"""Main pipeline orchestrator for the Adaptive Edge-AI Navigation System."""
import argparse
import sys
import os
import yaml
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from perception import CameraSource, FramePacket, FramePreprocessor

def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(base_dir, config_path)
        if os.path.exists(candidate):
            config_path = candidate
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_camera_pipeline(config: dict, source_override=None, max_frames: int = None, headless: bool = False) -> None:
    """Executes Step 2: Camera Acquisition, Validation, Timestamping, FPS measurement, and Preview."""
    cam_cfg = config.get("camera", {})
    source = source_override if source_override is not None else cam_cfg.get("source", 0)
    width = cam_cfg.get("width", 640)
    height = cam_cfg.get("height", 480)
    fps = cam_cfg.get("fps", 30)

    debug_cfg = config.get("debug", {})
    display_enabled = debug_cfg.get("display", True) and not headless
    window_name = debug_cfg.get("window_name", "Adaptive Navigation - Camera Stream")

    print("=" * 60)
    print("STEP 2: CAMERA ACQUISITION LAYER")
    print("=" * 60)
    print(f"Connecting to camera source: {source}")
    print(f"Target Resolution: {width}x{height} @ {fps} FPS")
    print(f"Debug Display: {'Enabled' if display_enabled else 'Disabled (Headless)'}")
    print("Press 'q' in preview window or Ctrl+C in terminal to stop.")
    print("-" * 60)

    preprocessor = FramePreprocessor()
    camera = CameraSource(source=source, width=width, height=height, target_fps=fps)

    try:
        camera.open()
    except Exception as e:
        print(f"\nERROR: Unable to open camera source: {e}")
        return

    frames_processed = 0

    try:
        while True:
            packet: FramePacket = camera.read_frame()

            if packet is None:
                # Video reached end or camera disconnected
                if camera.is_video_file:
                    print("\nEnd of video stream reached.")
                else:
                    print("\nWarning: Failed to capture frame from camera.")
                break

            # Frame validation
            if not preprocessor.validate_frame(packet.frame):
                print(f"Warning: Dropped invalid frame at index {packet.frame_index}")
                continue

            frames_processed += 1

            # Log frame capture progress periodically in console
            if frames_processed % 30 == 0 or frames_processed == 1:
                print(
                    f"[Frame {packet.frame_index:05d}] "
                    f"Res: {packet.resolution[0]}x{packet.resolution[1]} | "
                    f"Capture FPS: {packet.capture_fps:5.1f} | "
                    f"Timestamp: {packet.timestamp:12.3f}s"
                )

            # Debug visual overlay
            if display_enabled:
                display_frame = packet.frame.copy()
                overlay_text_1 = f"Frame: {packet.frame_index} | Res: {packet.resolution[0]}x{packet.resolution[1]}"
                overlay_text_2 = f"Capture FPS: {packet.capture_fps:.1f} | T: {packet.timestamp:.2f}s"
                
                cv2.putText(display_frame, overlay_text_1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, overlay_text_2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nUser requested exit via 'q'.")
                    break

            if max_frames is not None and frames_processed >= max_frames:
                print(f"\nReached maximum requested frame count ({max_frames}). Stopping.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")
    finally:
        # Resource cleanup
        camera.release()
        if display_enabled:
            cv2.destroyAllWindows()
        print("-" * 60)
        print(f"Camera released. Total frames captured: {frames_processed}")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Adaptive Edge-AI Navigation Prototype - Step 2")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--video", type=str, default=None, help="Path to test video file")
    parser.add_argument("--cam", type=int, default=None, help="Camera index")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames captured (for tests)")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI preview window")
    args = parser.parse_args()

    config = load_config(args.config)
    source = args.cam if args.cam is not None else args.video

    run_camera_pipeline(
        config=config,
        source_override=source,
        max_frames=args.max_frames,
        headless=args.headless,
    )

if __name__ == "__main__":
    main()
