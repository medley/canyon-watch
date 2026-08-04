"""Canyon Watch: poll UDOT SR-210 (Little Cottonwood Canyon) cameras,
count vehicles with local Roboflow inference, append a time series.

Run once per invocation (cron does the scheduling):
  .venv/bin/python canyon_watch.py

Outputs:
  canyon_watch/readings.jsonl          one line per camera per poll
  canyon_watch/snapshots/<ts>/<cam>.jpg  annotated snapshot per camera
"""

import json
import os
import urllib.request
from datetime import datetime, timezone

import torch

if not hasattr(torch.mps, "current_device"):
    torch.mps.current_device = lambda: 0

import cv2
import numpy as np
import supervision as sv
from dotenv import load_dotenv
from inference import get_model

BASE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE, ".env"))

CAMERAS = {
    "sr209-intersection": 141829,
    "upper-vault-mp5.96": 142227,
    "seven-turns-mp7.4": 136332,
    "white-pine-mp8.7": 142232,
    "upper-white-pine-mp9.7": 140390,
    "alta-bypass-mp10.95": 142061,
    "alta-mp12.16": 137940,
}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
OUT = os.path.join(BASE, "canyon_watch")

model = get_model("rfdetr-base", api_key=os.environ["ROBOFLOW_API_KEY"])
box_annotator = sv.BoxAnnotator(thickness=2)

ts = datetime.now(timezone.utc)
ts_label = ts.strftime("%Y%m%dT%H%M%SZ")
snap_dir = os.path.join(OUT, "snapshots", ts_label)
os.makedirs(snap_dir, exist_ok=True)

readings = []
for name, cam_id in CAMERAS.items():
    url = f"https://www.udottraffic.utah.gov/map/Cctv/{cam_id}"
    try:
        raw = urllib.request.urlopen(url, timeout=30).read()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("undecodable image")
        # night / dead-feed guard: nearly-black frames aren't countable
        if img.mean() < 20:
            readings.append({"ts": ts.isoformat(), "camera": name, "camera_id": cam_id,
                             "vehicles": None, "status": "dark"})
            continue
        dets = sv.Detections.from_inference(model.infer(img, confidence=0.25)[0])
        # RF-DETR can emit the same object under two classes (car + truck);
        # class-agnostic NMS keeps one box per physical vehicle
        dets = dets.with_nms(threshold=0.5, class_agnostic=True)
        mask = np.isin(dets.data.get("class_name", np.array([])), list(VEHICLE_CLASSES))
        vehicles = dets[mask]
        annotated = box_annotator.annotate(img.copy(), vehicles)
        cv2.putText(annotated, f"{name}  vehicles={len(vehicles)}  {ts_label}",
                    (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(snap_dir, f"{name}.jpg"), annotated)
        readings.append({"ts": ts.isoformat(), "camera": name, "camera_id": cam_id,
                         "vehicles": int(len(vehicles)), "status": "ok"})
    except Exception as e:
        readings.append({"ts": ts.isoformat(), "camera": name, "camera_id": cam_id,
                         "vehicles": None, "status": f"error: {e}"})

counts = [r["vehicles"] for r in readings if r["vehicles"] is not None]
busyness = int(sum(counts)) if counts else None

with open(os.path.join(OUT, "readings.jsonl"), "a") as f:
    for r in readings:
        f.write(json.dumps(r) + "\n")
    f.write(json.dumps({"ts": ts.isoformat(), "camera": "_canyon_total",
                        "vehicles": busyness, "status": f"{len(counts)}/7 cams ok"}) + "\n")

print(f"{ts_label} total={busyness} " +
      " ".join(f"{r['camera']}={r['vehicles']}" for r in readings))
