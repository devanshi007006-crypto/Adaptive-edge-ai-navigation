# Validation Videos Inventory

This directory contains controlled recorded test videos for Phase 2A behavioral validation of the Adaptive Edge-AI Navigation system.

## Directory Structure & Video Manifest

| Relative Path | Scenario / Category | Resolution | FPS | Frames | Duration (s) | File Size (MB) | Decode Status | Notes |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `approaching/S03_approaching_r01.mp4` | `approaching` | 848x478 | 59.17 | 788 | 13.32 | 2.59 | SUCCESS | Dynamic approaching obstacle / closing motion |
| `clear_path/S01_clear_r01.mp4` | `clear_path` | 848x478 | 57.65 | 1524 | 26.43 | 5.14 | SUCCESS | Negative control: uninhibited corridor navigation |
| `crossing/S05_crossing_r01.mp4` | `crossing` | 848x478 | 58.45 | 908 | 15.53 | 3.04 | SUCCESS | Orthogonal/transverse crossing pedestrian |
| `receding/S04_receding_r01.mp4` | `receding` | 848x478 | 59.31 | 857 | 14.45 | 2.81 | SUCCESS | Object moving away from camera |
| `static_obstacle/S02_static_r01.mp4` | `static_obstacle` | 848x478 | 59.18 | 1158 | 19.57 | 3.81 | SUCCESS | Stationary obstacle with ego-camera approach |

## Summary
- **Total Videos**: 5
- **Total Frames**: 5,235
- **Total Duration**: ~89.3 seconds
- **Resolution**: 848x478 (all videos)
- **Framerate**: ~58–59 FPS variable/recorded framerate
- **OpenCV Decode Health**: 100% (5/5 videos opened, all reported frames successfully read and verified)
