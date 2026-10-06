"""Canyon Watch: poll UDOT SR-210 (Little Cottonwood Canyon) cameras,
count vehicles with local Roboflow inference, append a time series.

Run once per invocation (cron does the scheduling):
  .venv/bin/python canyon_watch.py

Outputs:
  canyon_watch/readings.jsonl          one line per camera per poll
  canyon_watch/snapshots/<ts>/<cam>.jpg  annotated snapshot per camera
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone

import cv2
import numpy as np
import supervision as sv

BASE = os.path.dirname(os.path.abspath(__file__))

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
# sample_detection.jpg is 437033 bytes (1280x720). Cap a download at 10 MiB.
MAX_IMAGE_BYTES = 10 * 1024 * 1024

box_annotator = sv.BoxAnnotator(thickness=2)


def load_model():
    import torch

    if not hasattr(torch.mps, "current_device"):
        torch.mps.current_device = lambda: 0

    from inference import get_model

    return get_model("rfdetr-base", api_key=os.environ["ROBOFLOW_API_KEY"])


def fetch_image(url, timeout=30, max_bytes=MAX_IMAGE_BYTES) -> np.ndarray:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        raw = resp.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"image exceeds {max_bytes} bytes")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("undecodable image")
    return img


def select_vehicles(dets: sv.Detections) -> sv.Detections:
    # Drop non-vehicles before class-agnostic NMS so a person cannot delete a car.
    # RF-DETR can emit the same object under two classes (car + truck); the
    # class-agnostic NMS then keeps one box per physical vehicle.
    if len(dets) == 0:
        return dets
    names = (dets.data or {}).get("class_name")
    if names is None or len(np.asarray(names)) != len(dets):
        # Uncountable, not zero: read_camera records this as an error row.
        raise ValueError("inference result missing class_name")
    vehicles = dets[np.isin(np.asarray(names), list(VEHICLE_CLASSES))]
    if len(vehicles) > 1:
        vehicles = vehicles.with_nms(threshold=0.5, class_agnostic=True)
    return vehicles


def read_camera(name, cam_id, ts, ts_label, model, snap_dir, fetch=fetch_image) -> dict:
    url = f"https://www.udottraffic.utah.gov/map/Cctv/{cam_id}"
    try:
        img = fetch(url)
        # night / dead-feed guard: nearly-black frames aren't countable
        if img.mean() < 20:
            return {"ts": ts.isoformat(), "camera": name, "camera_id": cam_id,
                    "vehicles": None, "status": "dark"}
        dets = sv.Detections.from_inference(model.infer(img, confidence=0.25)[0])
        vehicles = select_vehicles(dets)
        annotated = box_annotator.annotate(img.copy(), vehicles)
        cv2.putText(annotated, f"{name}  vehicles={len(vehicles)}  {ts_label}",
                    (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(snap_dir, f"{name}.jpg"), annotated)
        return {"ts": ts.isoformat(), "camera": name, "camera_id": cam_id,
                "vehicles": int(len(vehicles)), "status": "ok"}
    except Exception as e:
        return {"ts": ts.isoformat(), "camera": name, "camera_id": cam_id,
                "vehicles": None, "status": f"error: {e}"}


def canyon_total(readings, ts, n_cams) -> dict:
    counts = [r["vehicles"] for r in readings if r["vehicles"] is not None]
    busyness = int(sum(counts)) if counts else None
    return {"ts": ts.isoformat(), "camera": "_canyon_total",
            "vehicles": busyness, "status": f"{len(counts)}/{n_cams} cams ok"}


def append_readings(path, readings, total):
    with open(path, "a") as f:
        for r in readings:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps(total) + "\n")


def run(model, out_dir, now=None, fetch=fetch_image, cameras=CAMERAS) -> list[dict]:
    ts = now if now is not None else datetime.now(timezone.utc)
    ts_label = ts.strftime("%Y%m%dT%H%M%SZ")
    snap_dir = os.path.join(out_dir, "snapshots", ts_label)
    os.makedirs(snap_dir, exist_ok=True)
    readings = []
    for name, cam_id in cameras.items():
        readings.append(
            read_camera(name, cam_id, ts, ts_label, model, snap_dir, fetch=fetch))
    total = canyon_total(readings, ts, len(cameras))
    append_readings(os.path.join(out_dir, "readings.jsonl"), readings, total)
    busyness = total["vehicles"]
    print(f"{ts_label} total={busyness} " +
          " ".join(f"{r['camera']}={r['vehicles']}" for r in readings))
    return readings


def main():
    from dotenv import load_dotenv

    load_dotenv(os.path.join(BASE, ".env"))
    run(load_model(), OUT)


if __name__ == "__main__":
    main()
