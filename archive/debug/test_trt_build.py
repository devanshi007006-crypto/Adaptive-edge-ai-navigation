import os
import sys
import time
from pathlib import Path
import numpy as np
import tensorrt as trt

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_MODELS_DIR = REPO_ROOT / "models/deployment"
DETECTOR_ONNX = REPO_ROOT / "models/detector/yolo11n.onnx"
DEPLOY_YOLO_ONNX = DEPLOY_MODELS_DIR / "yolo11n.onnx"
DEPTH_ONNX = DEPLOY_MODELS_DIR / "depth_anything_v2_vits.onnx"

# Ensure yolo11n.onnx is in deployment folder
if DETECTOR_ONNX.exists() and not DEPLOY_YOLO_ONNX.exists():
    import shutil
    shutil.copy(DETECTOR_ONNX, DEPLOY_YOLO_ONNX)
    print(f"Copied {DETECTOR_ONNX} to {DEPLOY_YOLO_ONNX}")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path, is_dynamic=False, dynamic_input_name="images", fixed_shape=(1, 3, 640, 640)):
    print(f"\n--- Building TensorRT FP16 Engine for {onnx_path.name} ---")
    t0 = time.perf_counter()
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error_idx in range(parser.num_errors):
                print(f"ONNX Parser Error: {parser.get_error(error_idx)}")
            raise RuntimeError(f"Failed to parse ONNX file: {onnx_path}")
            
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30) # 2GB workspace
    
    config.set_flag(trt.BuilderFlag.FP16)
    
    if is_dynamic:
        profile = builder.create_optimization_profile()
        profile.set_shape(dynamic_input_name, fixed_shape, fixed_shape, fixed_shape)
        config.add_optimization_profile(profile)
        
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError(f"Failed to build TensorRT engine for {onnx_path}")
        
    build_time = time.perf_counter() - t0
    
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
        
    print(f"Successfully built: {engine_path.name}")
    print(f"  Build time: {build_time:.2f} s")
    print(f"  Engine size: {len(serialized_engine) / (1024**2):.2f} MB")
    return build_time, len(serialized_engine)

if __name__ == "__main__":
    yolo_engine_path = DEPLOY_MODELS_DIR / "yolo11n_fp16.engine"
    depth_engine_path = DEPLOY_MODELS_DIR / "depth_anything_v2_vits_fp16.engine"
    
    # 1. Build YOLO11n Engine
    t_yolo, sz_yolo = build_engine(DEPLOY_YOLO_ONNX, yolo_engine_path, is_dynamic=True, dynamic_input_name="images", fixed_shape=(1, 3, 640, 640))
    
    # 2. Build Depth Anything V2 Engine
    t_depth, sz_depth = build_engine(DEPTH_ONNX, depth_engine_path, is_dynamic=False)
    
    print("\nAll TensorRT FP16 engines built successfully!")
