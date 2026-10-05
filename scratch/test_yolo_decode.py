import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import cv2
import torchvision
import tensorrt as trt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEPLOY_MODELS_DIR = REPO_ROOT / "models/deployment"

from adaptive_navigation.perception.detector import YOLOObjectDetector

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class NativeTRTEngine:
    def __init__(self, engine_path):
        self.runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        
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
            
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return output_tensors

def preprocess_yolo(frame, target_size=(640, 640)):
    h, w = frame.shape[:2]
    img_resized = cv2.resize(frame, target_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return img_tensor.cuda()

def decode_yolo(output0, orig_h, orig_w, conf_thres=0.25, iou_thres=0.45):
    preds = output0[0].float().transpose(0, 1) # (8400, 84)
    boxes = preds[:, :4] # cx, cy, w, h
    scores = preds[:, 4:] # (8400, 80)
    
    max_scores, class_ids = torch.max(scores, dim=1)
    mask = max_scores > conf_thres
    
    if not mask.any():
        return []
        
    boxes = boxes[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]
    
    # cx, cy, w, h (in 640x640) -> x1, y1, x2, y2
    x1 = (boxes[:, 0] - boxes[:, 2] / 2) * (orig_w / 640.0)
    y1 = (boxes[:, 1] - boxes[:, 3] / 2) * (orig_h / 640.0)
    x2 = (boxes[:, 0] + boxes[:, 2] / 2) * (orig_w / 640.0)
    y2 = (boxes[:, 1] + boxes[:, 3] / 2) * (orig_h / 640.0)
    boxes_corner = torch.stack([x1, y1, x2, y2], dim=1)
    
    keep = torchvision.ops.nms(boxes_corner, max_scores, iou_thres)
    
    results = []
    for idx in keep:
        results.append({
            "bbox": [float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])],
            "confidence": float(max_scores[idx]),
            "class_id": int(class_ids[idx])
        })
    return results

if __name__ == "__main__":
    video_path = REPO_ROOT / "validation/datasets/heads_up/sequences/HU_U01_multiped/HU_U01_multiped.mp4"
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    
    detector_py = YOLOObjectDetector(model_name_or_path="models/detector/yolo11n.pt", device="cuda:0")
    dets_py = detector_py.detect(frame, timestamp=0.0)
    
    yolo_engine = NativeTRTEngine(DEPLOY_MODELS_DIR / "yolo11n_fp16.engine")
    img_tensor = preprocess_yolo(frame)
    out_trt = yolo_engine.infer({"images": img_tensor})
    dets_trt = decode_yolo(out_trt["output0"], frame.shape[0], frame.shape[1])
    
    print(f"PyTorch YOLO11n Detections Count: {len(dets_py)}")
    for d in dets_py:
        print(f"  PyTorch: cls={d.class_id}, conf={d.confidence:.4f}, bbox={[round(x,1) for x in d.bbox]}")
        
    print(f"\nNative TRT YOLO11n Detections Count: {len(dets_trt)}")
    for d in dets_trt:
        print(f"  Native TRT: cls={d['class_id']}, conf={d['confidence']:.4f}, bbox={[round(x,1) for x in d['bbox']]}")
