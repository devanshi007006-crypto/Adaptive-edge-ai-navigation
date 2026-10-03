"""Main pipeline orchestrator for the Adaptive Edge-AI Navigation System."""
import argparse
import time
import sys
import os
import yaml
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from perception import CameraSource, FramePacket, FramePreprocessor, YOLOObjectDetector, Detection

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

def run_detection_pipeline(
    config: dict,
    source_override=None,
    model_override=None,
    max_frames: int = None,
    headless: bool = False,
) -> None:
    """Executes Step 3: Camera Acquisition -> Frame Validation -> YOLO Object Detection -> Visualization."""
    cam_cfg = config.get("camera", {})
    source = source_override if source_override is not None else cam_cfg.get("source", 0)
    width = cam_cfg.get("width", 640)
    height = cam_cfg.get("height", 480)
    fps = cam_cfg.get("fps", 30)

    det_cfg = config.get("detector", {})
    model_name = model_override if model_override is not None else det_cfg.get("model", "yolo11n.pt")
    conf_thresh = det_cfg.get("confidence_threshold", 0.25)
    iou_thresh = det_cfg.get("iou_threshold", 0.45)
    img_size = det_cfg.get("image_size", 640)
    device = det_cfg.get("device", "auto")

    debug_cfg = config.get("debug", {})
    display_enabled = debug_cfg.get("display", True) and not headless
    window_name = debug_cfg.get("window_name", "Adaptive Navigation - YOLO Detection")

    print("=" * 65)
    print("STEP 3: CAMERA + YOLO OBJECT DETECTION PIPELINE")
    print("=" * 65)
    print(f"Camera Source: {source} ({width}x{height} @ {fps} FPS)")
    print(f"YOLO Model:    {model_name} | Conf: {conf_thresh} | IoU: {iou_thresh} | ImgSz: {img_size}")
    print(f"Device Setting: {device}")
    print(f"Display Mode:  {'Active Window' if display_enabled else 'Headless'}")
    print("Press 'q' in preview window or Ctrl+C in terminal to stop.")
    print("-" * 65)

    # Initialize Camera
    preprocessor = FramePreprocessor()
    camera = CameraSource(source=source, width=width, height=height, target_fps=fps)
    try:
        camera.open()
    except Exception as e:
        print(f"\nERROR: Unable to open camera source: {e}")
        return

    # Initialize Detector
    try:
        detector = YOLOObjectDetector(
            model_name_or_path=model_name,
            confidence_threshold=conf_thresh,
            iou_threshold=iou_thresh,
            image_size=img_size,
            device=device,
        )
    except Exception as e:
        camera.release()
        print(f"\nERROR: Failed to initialize detector: {e}")
        return

    frames_processed = 0
    loop_times = []
    last_loop_time = time.perf_counter()

    try:
        while True:
            t_loop_start = time.perf_counter()
            packet: FramePacket = camera.read_frame()

            if packet is None:
                if camera.is_video_file:
                    print("\nEnd of video stream reached.")
                else:
                    print("\nWarning: Failed to capture frame from camera.")
                break

            # Frame validation
            if not preprocessor.validate_frame(packet.frame):
                print(f"Warning: Dropped invalid frame at index {packet.frame_index}")
                continue

            # Run YOLO Object Detection
            detections = detector.detect(packet.frame, packet.timestamp)
            inference_latency_ms = detector.last_inference_latency_ms

            frames_processed += 1

            # Measure overall loop FPS
            t_now = time.perf_counter()
            dt_loop = t_now - last_loop_time
            last_loop_time = t_now
            instant_loop_fps = (1.0 / dt_loop) if dt_loop > 0 else 0.0
            loop_times.append(instant_loop_fps)
            if len(loop_times) > 30:
                loop_times.pop(0)
            overall_fps = sum(loop_times) / len(loop_times)

            # Periodic console report
            if frames_processed % 30 == 0 or frames_processed == 1:
                class_counts = {}
                for d in detections:
                    class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
                det_str = ", ".join(f"{c}:{n}" for c, n in class_counts.items()) if class_counts else "None"
                print(
                    f"[Frame {packet.frame_index:05d}] "
                    f"Cam FPS: {packet.capture_fps:4.1f} | "
                    f"Loop FPS: {overall_fps:4.1f} | "
                    f"Inference: {inference_latency_ms:5.1f}ms | "
                    f"Objects ({len(detections)}): [{det_str}]"
                )

            # Debug visual overlay
            if display_enabled:
                display_frame = packet.frame.copy()
                for det in detections:
                    x1, y1, x2, y2 = [int(v) for v in det.bbox]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{det.class_name} {det.confidence:.2f}"
                    cv2.putText(
                        display_frame,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )
                    # Draw center point
                    cv2.circle(display_frame, (int(det.center_x), int(det.center_y)), 4, (0, 0, 255), -1)

                overlay_top = f"Frame: {packet.frame_index} | Detections: {len(detections)} | Cam FPS: {packet.capture_fps:.1f}"
                overlay_sub = f"Loop FPS: {overall_fps:.1f} | Inference Latency: {inference_latency_ms:.1f}ms"
                cv2.putText(display_frame, overlay_top, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                cv2.putText(display_frame, overlay_sub, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

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
        camera.release()
        if display_enabled:
            cv2.destroyAllWindows()
        print("-" * 65)
        print(f"Camera released. Total frames processed: {frames_processed}")
        print("=" * 65)

def main():
    parser = argparse.ArgumentParser(description="Adaptive Edge-AI Navigation - Step 3: YOLO Detection")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--video", type=str, default=None, help="Path to test video file")
    parser.add_argument("--cam", type=int, default=None, help="Camera index")
    parser.add_argument("--model", type=str, default=None, help="Override YOLO model checkpoint")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed (for testing)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI preview window")
    args = parser.parse_args()

    config = load_config(args.config)
    source = args.cam if args.cam is not None else args.video

    run_detection_pipeline(
        config=config,
        source_override=source,
        model_override=args.model,
        max_frames=args.max_frames,
        headless=args.headless,
    )

if __name__ == "__main__":
    main()
