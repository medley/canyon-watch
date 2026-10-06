"""Hermetic tests for canyon_watch. No network, torch, or Roboflow."""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import cv2
import numpy as np
import pytest
import supervision as sv

import canyon_watch as cw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NOW = datetime(2026, 8, 4, 19, 14, 22, tzinfo=timezone.utc)
LABEL = "20260804T191422Z"


def boxes(xyxy, confidence, names):
    xyxy = np.asarray(xyxy, dtype=np.float32)
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.asarray(confidence, dtype=np.float32),
        class_id=np.arange(len(xyxy)),
        data={"class_name": np.asarray(names)},
    )


def bright(value=180, shape=(32, 48, 3)):
    return np.full(shape, value, np.uint8)


class Model:
    def __init__(self, detections):
        self.detections = detections
        self.confidence = None

    def infer(self, img, confidence=0.25):
        self.confidence = confidence
        return [self.detections]


class Body:
    def __init__(self, payload, record):
        self.payload = payload
        self.record = record

    def read(self, n=-1):
        self.record.append(n)
        if n is None or n < 0:
            return self.payload
        return self.payload[:n]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def passthrough(monkeypatch):
    monkeypatch.setattr(
        sv.Detections,
        "from_inference",
        classmethod(lambda cls, result: result),
    )


def test_camera_table_is_unchanged():
    assert cw.CAMERAS == {
        "sr209-intersection": 141829,
        "upper-vault-mp5.96": 142227,
        "seven-turns-mp7.4": 136332,
        "white-pine-mp9.2": 142232,
        "upper-white-pine-mp9.7": 140390,
        "alta-bypass-mp10.95": 142061,
        "alta-mp12.16": 137940,
    }
    assert cw.VEHICLE_CLASSES == {"car", "truck", "bus", "motorcycle"}


def test_import_needs_no_torch_inference_or_api_key(tmp_path):
    # Importing must not load the model, read the key, or poll the cameras.
    # torch and inference are blocked outright, so any import-time use fails.
    code = (
        "import sys; sys.modules['torch'] = None; sys.modules['inference'] = None\n"
        "import canyon_watch\n"
        "print(sorted(canyon_watch.CAMERAS)[0])\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "ROBOFLOW_API_KEY"}
    env["PYTHONPATH"] = REPO
    out_dir = os.path.join(REPO, "canyon_watch")
    before = sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else None
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "alta-bypass-mp10.95\n"
    after = sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else None
    assert after == before


def test_main_runs_one_poll_into_out(monkeypatch):
    calls = []
    sentinel = object()
    monkeypatch.setattr(cw, "load_model", lambda: sentinel)
    monkeypatch.setattr(cw, "run", lambda model, out: calls.append((model, out)))
    cw.main()
    assert calls == [(sentinel, cw.OUT)]


def test_person_overlap_does_not_drop_the_car(passthrough, tmp_path):
    # Same box, IoU 1.0. Person 0.90 deletes car 0.40 if NMS runs first.
    found = boxes(
        [[0, 0, 100, 100], [0, 0, 100, 100]],
        [0.40, 0.90],
        ["car", "person"],
    )
    assert list(cw.select_vehicles(found).data["class_name"]) == ["car"]
    model = Model(found)
    snap = tmp_path / "s"
    snap.mkdir()
    reading = cw.read_camera(
        "white-pine-mp9.2", 142232, NOW, LABEL, model, str(snap),
        fetch=lambda url: bright(),
    )
    assert model.confidence == 0.25
    assert reading["vehicles"] == 1
    assert reading["status"] == "ok"
    assert reading["camera_id"] == 142232
    assert (snap / "white-pine-mp9.2.jpg").is_file()


def test_two_labels_on_one_object_count_as_one():
    # README pair: car 0.40 and truck 0.46 on one box.
    found = boxes(
        [[10, 10, 90, 70], [10, 10, 90, 70]],
        [0.40, 0.46],
        ["car", "truck"],
    )
    assert len(cw.select_vehicles(found)) == 1


def test_nms_threshold_stays_at_half():
    # 0.40 is confidence. The 0.5 cutoff is IoU.
    # Height 40 inside a 100-tall box: intersection/union = 0.4, both stay.
    under = boxes(
        [[0, 0, 100, 100], [0, 0, 100, 40]],
        [0.40, 0.90],
        ["car", "truck"],
    )
    assert len(cw.select_vehicles(under)) == 2
    # Height 60: intersection/union = 0.6, one box remains.
    over = boxes(
        [[0, 0, 100, 100], [0, 0, 100, 60]],
        [0.40, 0.46],
        ["car", "truck"],
    )
    assert len(cw.select_vehicles(over)) == 1


def test_non_vehicle_is_dropped():
    found = boxes(
        [[0, 0, 10, 10], [40, 40, 55, 70]],
        [0.99, 0.40],
        ["person", "motorcycle"],
    )
    kept = cw.select_vehicles(found)
    assert list(kept.data["class_name"]) == ["motorcycle"]


def test_empty_and_missing_class_name_do_not_raise():
    empty = sv.Detections(
        xyxy=np.zeros((0, 4), np.float32),
        confidence=np.zeros((0,), np.float32),
        class_id=np.zeros((0,), np.int64),
        data={"class_name": np.array([])},
    )
    assert len(cw.select_vehicles(empty)) == 0
    bare = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]], np.float32),
        confidence=np.array([0.95], np.float32),
        class_id=np.array([1]),
    )
    with pytest.raises(ValueError, match="missing class_name"):
        cw.select_vehicles(bare)


def test_missing_class_name_is_an_error_row_not_zero(passthrough, tmp_path):
    bare = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]], np.float32),
        confidence=np.array([0.95], np.float32),
        class_id=np.array([1]),
    )
    reading = cw.read_camera(
        "alta-mp12.16", 137940, NOW, LABEL, Model(bare), str(tmp_path),
        fetch=lambda url: bright(),
    )
    assert reading["vehicles"] is None
    assert reading["status"] == "error: inference result missing class_name"


def test_dark_frame_is_null_and_skips_inference(tmp_path):
    model = Model(boxes([[0, 0, 5, 5]], [0.9], ["car"]))
    snap = tmp_path / "s"
    snap.mkdir()
    reading = cw.read_camera(
        "alta-mp12.16", 137940, NOW, LABEL, model, str(snap),
        fetch=lambda url: bright(0),
    )
    assert reading["vehicles"] is None
    assert reading["status"] == "dark"
    assert model.confidence is None
    assert list(snap.iterdir()) == []


def test_mean_of_20_counts_as_zero_vehicles(passthrough, tmp_path):
    empty = sv.Detections(
        xyxy=np.zeros((0, 4), np.float32),
        confidence=np.zeros((0,), np.float32),
        class_id=np.zeros((0,), np.int64),
        data={"class_name": np.array([])},
    )
    model = Model(empty)
    snap = tmp_path / "s"
    snap.mkdir()
    reading = cw.read_camera(
        "alta-bypass-mp10.95", 142061, NOW, LABEL, model, str(snap),
        fetch=lambda url: bright(20),
    )
    assert reading["vehicles"] == 0
    assert reading["status"] == "ok"
    assert model.confidence == 0.25
    assert (snap / "alta-bypass-mp10.95.jpg").is_file()


def test_error_on_one_camera_keeps_a_partial_sum(passthrough, tmp_path, capsys):
    def fetch(url):
        if url.endswith("/2"):
            raise ValueError("undecodable image")
        return bright()

    readings = cw.run(
        Model(boxes([[0, 0, 8, 8]], [0.88], ["car"])),
        str(tmp_path),
        now=NOW,
        fetch=fetch,
        cameras={"sr209-intersection": 1, "seven-turns-mp7.4": 2},
    )
    assert [r["vehicles"] for r in readings] == [1, None]
    assert readings[1]["status"] == "error: undecodable image"
    lines = (tmp_path / "readings.jsonl").read_text().splitlines()
    assert len(lines) == 3
    assert list(json.loads(lines[0])) == [
        "ts", "camera", "camera_id", "vehicles", "status",
    ]
    total = json.loads(lines[-1])
    assert list(total) == ["ts", "camera", "vehicles", "status"]
    assert total["vehicles"] == 1
    assert total["status"] == "1/2 cams ok"
    assert capsys.readouterr().out == (
        "20260804T191422Z total=1 sr209-intersection=1 seven-turns-mp7.4=None\n"
    )


def test_total_uses_n_cams_and_keeps_partial_sums():
    partial = [{"vehicles": 10}] * 6 + [{"vehicles": None}]
    total = cw.canyon_total(partial, NOW, n_cams=7)
    assert total["vehicles"] == 60
    assert total["status"] == "6/7 cams ok"
    short = cw.canyon_total(
        [{"vehicles": 3}, {"vehicles": None}, {"vehicles": 4}], NOW, n_cams=3,
    )
    assert short["vehicles"] == 7
    assert short["status"] == "2/3 cams ok"
    zeros = cw.canyon_total([{"vehicles": 0}, {"vehicles": 0}], NOW, n_cams=2)
    assert zeros["vehicles"] == 0
    assert zeros["status"] == "2/2 cams ok"
    none = cw.canyon_total(
        [{"vehicles": None}, {"vehicles": None}], NOW, n_cams=4,
    )
    assert none["vehicles"] is None
    assert none["status"] == "0/4 cams ok"


def test_run_appends_and_skips_dark_snapshots(passthrough, tmp_path, capsys):
    cameras = {"upper-white-pine-mp9.7": 140390, "alta-mp12.16": 137940}

    def fetch(url):
        if url.endswith("137940"):
            return bright(0)
        return bright()

    model = Model(boxes([[1, 1, 9, 9]], [0.5], ["motorcycle"]))
    cw.run(model, str(tmp_path), now=NOW, fetch=fetch, cameras=cameras)
    cw.run(model, str(tmp_path), now=NOW, fetch=fetch, cameras=cameras)
    lines = (tmp_path / "readings.jsonl").read_text().splitlines()
    assert len(lines) == 6
    rows = [json.loads(line) for line in lines[:3]]
    assert rows[0]["vehicles"] == 1 and rows[0]["status"] == "ok"
    assert rows[1]["vehicles"] is None and rows[1]["status"] == "dark"
    assert rows[2]["vehicles"] == 1 and rows[2]["status"] == "1/2 cams ok"
    assert "camera_id" not in rows[2]
    snap = tmp_path / "snapshots" / LABEL
    assert (snap / "upper-white-pine-mp9.7.jpg").is_file()
    assert not (snap / "alta-mp12.16.jpg").exists()
    assert capsys.readouterr().out == (
        "20260804T191422Z total=1 upper-white-pine-mp9.7=1 alta-mp12.16=None\n"
        "20260804T191422Z total=1 upper-white-pine-mp9.7=1 alta-mp12.16=None\n"
    )


def test_fetch_reads_one_past_the_cap(monkeypatch):
    seen = []

    def urlopen(url, timeout=30):
        assert timeout == 30
        return Body(b"x" * 50, seen)

    monkeypatch.setattr(cw.urllib.request, "urlopen", urlopen)
    with pytest.raises(ValueError, match=r"^image exceeds 16 bytes$"):
        cw.fetch_image(
            "https://www.udottraffic.utah.gov/map/Cctv/141829", max_bytes=16,
        )
    assert seen == [17]


def test_body_at_the_cap_is_decoded(monkeypatch):
    img = bright()
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    payload = buf.tobytes()
    seen = []

    def urlopen(url, timeout=30):
        return Body(payload, seen)

    monkeypatch.setattr(cw.urllib.request, "urlopen", urlopen)
    out = cw.fetch_image("https://example.test/frame", max_bytes=len(payload))
    assert seen == [len(payload) + 1]
    assert out.shape == img.shape


def test_default_fetch_rejects_an_oversize_camera(monkeypatch, tmp_path):
    seen = []
    opened = {}

    def urlopen(url, timeout=30):
        opened["url"] = url
        opened["timeout"] = timeout
        return Body(b"z" * (cw.MAX_IMAGE_BYTES + 1), seen)

    monkeypatch.setattr(cw.urllib.request, "urlopen", urlopen)
    reading = cw.read_camera(
        "alta-mp12.16", 137940, NOW, LABEL, None, str(tmp_path),
    )
    assert opened == {
        "url": "https://www.udottraffic.utah.gov/map/Cctv/137940",
        "timeout": 30,
    }
    assert seen == [cw.MAX_IMAGE_BYTES + 1]
    assert reading["vehicles"] is None
    assert reading["status"] == (
        f"error: image exceeds {cw.MAX_IMAGE_BYTES} bytes"
    )
    assert list(tmp_path.iterdir()) == []
