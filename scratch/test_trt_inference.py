import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import cv2
import tensorrt as trt

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_MODELS_DIR = REPO_ROOT / "models/deployment"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class NativeTRTEngine:
    def __init__(self, engine_path):
        self.runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = [name for name in self.tensor_names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
        self.output_names = [name for name in self.tensor_names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]
        
    def infer(self, input_tensors_dict):
        bindings = {}
        output_tensors = {}
        
        for name in self.input_names:
            inp = input_tensors_dict[name]
            if not inp.is_cuda:
                inp = inp.cuda()
            inp = inp.contiguous()
            self.context.set_input_shape(name, inp.shape)
            self.context.set_tensor_address(name, inp.data_ptr())
            bindings[name] = inp
            
        for name in self.output_names:
            out_shape = self.context.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = torch.float32
            if dtype == trt.DataType.HALF:
                torch_dtype = torch.float16
            elif dtype == trt.DataType.INT32:
                torch_dtype = torch.int32
            out_tensor = torch.empty(tuple(out_shape), dtype=torch_dtype, device="cuda:0")
            self.context.set_tensor_address(name, out_tensor.data_ptr())
            output_tensors[name] = out_tensor
            
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return output_tensors

if __name__ == "__main__":
    print("Testing Native TensorRT Inference on RTX 4050...")
    yolo_engine = NativeTRTEngine(DEPLOY_MODELS_DIR / "yolo11n_fp16.engine")
    depth_engine = NativeTRTEngine(DEPLOY_MODELS_DIR / "depth_anything_v2_vits_fp16.engine")
    
    # Test YOLO11n
    dummy_img = torch.randn(1, 3, 640, 640, device="cuda:0", dtype=torch.float32)
    t0 = time.perf_counter()
    out_yolo = yolo_engine.infer({"images": dummy_img})
    t_yolo = (time.perf_counter() - t0) * 1000.0
    print(f"YOLO11n TRT Latency: {t_yolo:.2f} ms | Output Shape: {out_yolo['output0'].shape}")
    
    # Test Depth Anything V2
    dummy_depth_input = torch.randn(1, 3, 518, 518, device="cuda:0", dtype=torch.float32)
    t0 = time.perf_counter()
    out_depth = depth_engine.infer({"input": dummy_depth_input})
    t_depth = (time.perf_counter() - t0) * 1000.0
    print(f"Depth Anything V2 TRT Latency: {t_depth:.2f} ms | Output Shape: {out_depth['output'].shape}")
    print("Native TensorRT execution test successful!")
