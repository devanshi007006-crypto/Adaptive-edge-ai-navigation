import json
import csv
from pathlib import Path

telemetry_json = Path(r"c:\My sep_stuffs\Research Conclave\Adaptive-edge-ai-navigation\validation\results\heads_up\HU_U01_multiped\telemetry.json")

if telemetry_json.exists():
    with open(telemetry_json, "r") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} telemetry records.")
    print("Record 0 keys:", list(records[0].keys()))
    print("Record 0 frame info:", records[0].get("frame_index"), records[0].get("timestamp"))
    print("Record 0 objects sample:", records[0].get("objects")[:1] if records[0].get("objects") else [])
else:
    print("Telemetry JSON not found.")
