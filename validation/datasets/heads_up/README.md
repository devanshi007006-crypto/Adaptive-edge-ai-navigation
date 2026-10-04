# HEADS-UP Egocentric Dataset Guide & Repository Hygiene

**Official Dataset**: HEADS-UP (*Head-Mounted Egocentric Dataset for Trajectory Prediction in Blind Assistance Systems*)  
**Authors**: Yassaman Haghighi, et al. (EPFL VITA Laboratory)  
**Publication**: [arXiv:2409.20324v1](https://arxiv.org/abs/2409.20324) (September 2024)  
**Repository Source**: Hugging Face Hub: [`Yassaman/HEADS-UP`](https://huggingface.co/datasets/Yassaman/HEADS-UP)  
**Evaluated Git Revision / Commit**: `e166f2641d737303e9854fbf4e2526ac75c39b21`  

---

## 1. Git Hygiene Policy & Ignored Large Artifacts

In compliance with repository hygiene rules:
- **Raw and large dataset artifacts are strictly LOCAL-ONLY and NEVER committed to Git.**
- The full ~102 GB dataset archive is prohibited from repository commits.
- All downloaded archives (`*.tar`, `*.tar.gz`, `*.zip`), raw staging folders (`raw/`, `cache/`), and extracted image frame sequences (`sequences/`) are permanently excluded via `.gitignore`.
- Only lightweight metadata (`metadata/`), adapters (`heads_up_adapter.py`), runner scripts, and markdown documentation/reports are tracked in version control.

### Excluded vs. Tracked Assets

| Category | Path | Version Controlled? | Purpose |
| :--- | :--- | :---: | :--- |
| **Raw Archives** | `validation/datasets/heads_up/*.tar.gz` | **NO (Ignored)** | Downloaded dataset tarballs |
| **Extracted Sequences** | `validation/datasets/heads_up/sequences/` | **NO (Ignored)** | Extracted uncompressed PNG frames & MP4s |
| **Temporary Caches** | `validation/datasets/heads_up/cache/` | **NO (Ignored)** | Intermediate working caches |
| **Adapter Code** | `validation/datasets/heads_up/heads_up_adapter.py` | **YES** | Standard data loading interface |
| **Metadata CSVs** | `validation/datasets/heads_up/metadata/` | **YES** | Sliced camera poses, trajectories, calibrations |
| **Validation Results** | `validation/results/heads_up/` | **YES** | Per-run JSON & CSV telemetry, reports |

---

## 2. Authentication Protocol

HEADS-UP is a **gated** research dataset on the Hugging Face Hub requiring authenticated access.

### Setting the Environment Variable
Credentials must **NEVER** be hardcoded in source code, scripts, or markdown files. Access is granted strictly via the `HF_TOKEN` environment variable.

#### Windows PowerShell:
```powershell
$env:HF_TOKEN = "your_huggingface_token_here"
```

#### Linux / macOS Bash:
```bash
export HF_TOKEN="your_huggingface_token_here"
```

The data access scripts (`scripts/tools/download_heads_up_unconstrained.py`, `scripts/tools/resume_heads_up_unconstrained.py`) read `os.environ.get("HF_TOKEN")` and fail immediately with an explicit error if the token is absent.

---

## 3. Reproducing the Representative Evaluation Sample

To reproduce the Phase 3A sample evaluation without storing credentials or downloading 102 GB:

1. **Set your token in the active terminal**:
   ```powershell
   $env:HF_TOKEN = "your_huggingface_token_here"
   ```

2. **Download the targeted unconstrained archive** (~11.5 GB):
   ```powershell
   .\.venv\Scripts\python.exe scripts/tools/download_heads_up_unconstrained.py
   ```

3. **Extract and finalize the representative episodes**:
   ```powershell
   .\.venv\Scripts\python.exe scripts/tools/finalize_heads_up_sequences.py
   ```
   This generates the 3 representative sequence directories under `sequences/`:
   - `HU_unconstrained_s01_multiped` (73 frames: multi-pedestrian density)
   - `HU_unconstrained_s02_approach` (102 frames: closing hazard approach)
   - `HU_unconstrained_s03_headmotion` (75 frames: severe head motion >60°/s)

4. **Reclaim archive storage**:
   ```powershell
   Remove-Item validation/datasets/heads_up/rgb_unconstrained.tar.gz
   ```

5. **Execute the real Phase 3A validation benchmark**:
   ```powershell
   .\.venv\Scripts\python.exe scripts/run/run_phase3a_heads_up_validation.py --device cuda:0 --depth-cadence 2
   ```

---

## 4. Dataset Provenance Limitation Statement

The pedestrian trajectory annotations in the HEADS-UP benchmark were generated using an offline automated machine pipeline:
1. YOLOv8x bounding-box detections.
2. ByteTrack spatial tracking.
3. 2.5 FPS temporal downsampling with 200 ms temporal smoothing.
4. Linear Kalman filtering and depth reprojection from stereo disparity.

They serve as **external machine reference pseudo-labels** for qualitative correlation, NOT independent ground-truth human annotations or motion-capture ground truth.
