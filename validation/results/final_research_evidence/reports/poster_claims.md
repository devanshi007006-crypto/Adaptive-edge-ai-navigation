# Research Claims Guidance & Scope Boundaries

> **Document ID:** `validation/results/final_research_evidence/reports/poster_claims.md`

---

## 1. SAFE TO CLAIM
- Real-time throughput (**17.68 FPS**, **50.13 ms** latency) on laptop mobile GPU (RTX 4050).
- Scale-invariant monocular Time-to-Collision ($	au = d/\dot{d}$) using relative disparity derivatives.
- 100% false warning elimination on receding/passing obstacles compared to static range sensors.
- 2D image-plane gait sway absorption using sparse optical flow background divergence.
- Non-blocking offline text-to-speech audio advisories.

## 2. QUALIFIED CLAIMS
- Metric distance is valid ONLY when active ground-plane affine calibration is enabled.
- Spatial navigation provides 2D walking corridor avoidance (`FORWARD`, `LEFT`, `RIGHT`, `STOP`), not 3D global path SLAM.

## 3. DO NOT CLAIM
- DO NOT claim certified medical device or clinical mobility aid status.
- DO NOT claim full 3D 6-DOF Visual-Inertial Odometry (VIO) (no IMU fusion).
- DO NOT claim closed-loop cloud learning or training loops (Layer 3 is future extension).
