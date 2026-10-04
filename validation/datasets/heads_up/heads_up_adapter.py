"""
HEADS-UP Egocentric Dataset Adapter.

Provides standardized access to head-mounted egocentric sequences from the
official HEADS-UP benchmark (Haghighi et al., arXiv:2409.20324v1, EPFL VITA Lab).

Design Constraints:
- Exposes raw 1920x1080 frames without silent resizing or modification.
- Monotonic timestamps aligned to the ZED Mini 30 FPS sensor clock (dt = 1/30s).
- Synchronizes frame images with 6-DOF camera poses and trajectory reference labels.
- Clear demarcation: Trajectory labels are external machine references (YOLOv8 + ByteTrack + Kalman),
  NOT independent ground truth.
"""

import os
import json
import csv
import glob
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator
import cv2
import numpy as np


@dataclass
class HEADSUpCameraPose:
    """6-DOF camera pose from ZED SDK Visual-Inertial Odometry."""
    x: float
    y: float
    z: float
    q1: float  # x
    q2: float  # y
    q3: float  # z
    q4: float  # w


@dataclass
class HEADSUpPedestrianReference:
    """External pedestrian trajectory reference annotation."""
    agent_id: int
    bbox_xywh: Tuple[float, float, float, float]
    coords_3d: Tuple[float, float, float]  # (x, y, z) in meters in camera frame


@dataclass
class HEADSUpFramePacket:
    """Complete synchronized data packet for a single egocentric frame."""
    frame_index: int                       # 0-indexed sequential frame in episode
    heads_up_frame_id: int                 # Original dataset global frame number
    timestamp: float                       # Monotonic timestamp in seconds (frame_index / 30.0)
    frame: np.ndarray                      # H x W x 3 uint8 BGR image (1920x1080)
    camera_pose: Optional[HEADSUpCameraPose] = None
    pedestrian_references: List[HEADSUpPedestrianReference] = field(default_factory=list)


@dataclass
class HEADSUpSequenceMetadata:
    """Metadata describing a selected HEADS-UP sequence episode."""
    sequence_id: str
    subset: str                           # 'easy', 'hard', 'unconstrained'
    start_frame_id: int
    end_frame_id: int
    num_frames: int
    fps: float = 30.0
    width: int = 1920
    height: int = 1080
    scenario_description: str = ""
    camera_model: str = "Stereolabs ZED Mini"
    mount_type: str = "Cap-mounted head rig"


class HEADSUpDatasetAdapter:
    """
    Adapter exposing HEADS-UP egocentric sequence frames to the perception pipeline.
    """

    def __init__(self, sequence_dir: str):
        """
        Initialize adapter from a sequence directory.
        
        Expected directory structure:
            sequence_dir/
                metadata.json
                frames/
                    left_00000000.png, ...
                camera_poses.csv (optional)
                trajectories.csv (optional)
        """
        self.sequence_dir = sequence_dir
        self.metadata_file = os.path.join(sequence_dir, "metadata.json")
        self.frames_dir = os.path.join(sequence_dir, "frames")
        
        if not os.path.exists(self.frames_dir):
            raise FileNotFoundError(f"Frames directory not found: {self.frames_dir}")
            
        self.metadata: Optional[HEADSUpSequenceMetadata] = None
        self._load_metadata()
        
        # Discover and sort all frame files
        pattern = os.path.join(self.frames_dir, "*.png")
        self.frame_paths = sorted(glob.glob(pattern))
        if not self.frame_paths:
            # Fallback to jpg or other extensions
            self.frame_paths = sorted(glob.glob(os.path.join(self.frames_dir, "*.*")))
            
        if not self.frame_paths:
            raise ValueError(f"No image files found in {self.frames_dir}")
            
        self.num_frames = len(self.frame_paths)
        
        # Load optional poses and trajectory references
        self.camera_poses: Dict[int, HEADSUpCameraPose] = {}
        self.pedestrian_refs: Dict[int, List[HEADSUpPedestrianReference]] = {}
        self._load_poses()
        self._load_trajectories()

    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.metadata = HEADSUpSequenceMetadata(**data)

    def _load_poses(self):
        pose_path = os.path.join(self.sequence_dir, "camera_poses.csv")
        if not os.path.exists(pose_path):
            return
        with open(pose_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                try:
                    fid = int(row[1].strip('"'))
                    pose = HEADSUpCameraPose(
                        x=float(row[2]), y=float(row[3]), z=float(row[4]),
                        q1=float(row[5]), q2=float(row[6]), q3=float(row[7]), q4=float(row[8])
                    )
                    self.camera_poses[fid] = pose
                except Exception:
                    pass

    def _load_trajectories(self):
        traj_path = os.path.join(self.sequence_dir, "trajectories.csv")
        if not os.path.exists(traj_path):
            return
        import ast
        with open(traj_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                try:
                    fid = int(row[1].strip('"'))
                    coords = ast.literal_eval(row[2])
                    refs = []
                    for c in coords:
                        aid = c.get('id', -1)
                        bbox = tuple(c.get('bbox', [0, 0, 0, 0]))
                        coords_3d = (c.get('x', 0.0), c.get('y', 0.0), c.get('z', 0.0))
                        refs.append(HEADSUpPedestrianReference(aid, bbox, coords_3d))
                    self.pedestrian_refs[fid] = refs
                except Exception:
                    pass

    def __len__(self) -> int:
        return self.num_frames

    def get_frame(self, index: int) -> HEADSUpFramePacket:
        """
        Get frame packet by sequential index (0 to num_frames - 1).
        """
        if index < 0 or index >= self.num_frames:
            raise IndexError(f"Frame index {index} out of range (0-{self.num_frames-1})")
            
        path = self.frame_paths[index]
        basename = os.path.splitext(os.path.basename(path))[0]
        # Extract global frame id if filename matches left_%08d
        import re
        m = re.search(r'(\d+)', basename)
        global_fid = int(m.group(1)) if m else index
        
        frame = cv2.imread(path)
        if frame is None:
            raise IOError(f"Failed to read image at {path}")
            
        timestamp = index / (self.metadata.fps if self.metadata else 30.0)
        pose = self.camera_poses.get(global_fid)
        refs = self.pedestrian_refs.get(global_fid, [])
        
        return HEADSUpFramePacket(
            frame_index=index,
            heads_up_frame_id=global_fid,
            timestamp=timestamp,
            frame=frame,
            camera_pose=pose,
            pedestrian_references=refs
        )

    def stream_frames(self) -> Iterator[HEADSUpFramePacket]:
        """Generator yielding each frame packet sequentially."""
        for i in range(self.num_frames):
            yield self.get_frame(i)

    def export_video(self, output_mp4: str, fps: float = 30.0) -> str:
        """
        Export image sequence to an MP4 video file for pipeline streaming.
        """
        if self.num_frames == 0:
            raise ValueError("No frames to export")
            
        first = cv2.imread(self.frame_paths[0])
        h, w = first.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
        writer = cv2.VideoWriter(output_mp4, fourcc, fps, (w, h))
        
        for path in self.frame_paths:
            img = cv2.imread(path)
            writer.write(img)
        writer.release()
        return output_mp4
