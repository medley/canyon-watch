"""Path tests for canyon_watch. No network, torch, or Roboflow."""

import json
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest
import supervision as sv

import canyon_watch as cw

NOW = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
NOW_TS = "2026-04-01T12:00:00+00:00"
NOW_LABEL = "20260401T120000Z"

PER_CAM_KEYS = ["ts", "camera", "camera_id", "vehicles", "status"]
TOTAL_KEYS = ["ts", "camera", "vehicles", "status"]

# COCO detection names. Vehicle names match cw.VEHICLE_CLASSES.
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


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


def empty_detections():
    return sv.Detections(
        xyxy=np.zeros((0, 4), np.float32),
        confidence=np.zeros((0,), np.float32),
        class_id=np.zeros((0,), np.int64),
        data={"class_name": np.array([])},
    )


def jpeg_bytes(img):
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


class Body:
    def __init__(self, payload):
        self.payload = payload

    def read(self, n=-1):
        if n is None or n < 0:
            return self.payload
        return self.payload[:n]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class RecordingModel:
    def __init__(self, detections):
        self.detections = detections
        self.calls = []

    def infer(self, img, confidence=0.25):
        self.calls.append((confidence, tuple(img.shape)))
        return [self.detections]


class Boom:
    def __init__(self):
        self.calls = []

    def infer(self, img, confidence=0.25):
        self.calls.append(confidence)
        raise RuntimeError("detector failed")


@pytest.fixture
def passthrough(monkeypatch):
    monkeypatch.setattr(
        sv.Detections,
        "from_inference",
        classmethod(lambda cls, result: result),
    )


def test_fetch_image_decodes_jpeg_and_passes_timeout(monkeypatch):
    img = np.full((20, 34, 3), 90, np.uint8)
    payload = jpeg_bytes(img)
    seen = {}

    def urlopen(url, timeout=30):
        seen["timeout"] = timeout
        return Body(payload)

    monkeypatch.setattr(cw.urllib.request, "urlopen", urlopen)
    out = cw.fetch_image("https://www.udottraffic.utah.gov/map/Cctv/141829", timeout=11)
    assert seen["timeout"] == 11
    assert out.shape == (20, 34, 3)
    assert out.dtype == np.uint8


def test_fetch_image_rejects_undecodable_bytes(monkeypatch):
    def urlopen(url, timeout=30):
        return Body(b"not-a-jpeg")

    monkeypatch.setattr(cw.urllib.request, "urlopen", urlopen)
    with pytest.raises(ValueError) as caught:
        cw.fetch_image("https://www.udottraffic.utah.gov/map/Cctv/141829")
    assert str(caught.value) == "undecodable image"


def test_select_vehicles_keeps_only_coco_vehicle_classes():
    assert len(COCO_CLASSES) == 80
    assert set(cw.VEHICLE_CLASSES) <= set(COCO_CLASSES)
    xyxy = [[i * 20, 0, i * 20 + 9, 9] for i in range(len(COCO_CLASSES))]
    found = boxes(xyxy, [0.55] * len(COCO_CLASSES), COCO_CLASSES)
    kept = cw.select_vehicles(found)
    assert sorted(str(name) for name in kept.data["class_name"]) == [
        "bus", "car", "motorcycle", "truck",
    ]
    assert len(kept) == 4


def test_two_separate_cars_are_both_kept():
    found = boxes(
        [[0, 0, 40, 30], [80, 10, 130, 50]],
        [0.33, 0.81],
        ["car", "car"],
    )
    kept = cw.select_vehicles(found)
    assert len(kept) == 2
    assert [str(name) for name in kept.data["class_name"]] == ["car", "car"]
    got = kept.xyxy[np.argsort(kept.xyxy[:, 0])]
    expected = np.array([[0, 0, 40, 30], [80, 10, 130, 50]], np.float32)
    assert np.array_equal(got, expected)


def test_nms_keeps_the_higher_confidence_box():
    # Same box: car 0.30 loses to truck 0.80. The other two boxes do not overlap.
    found = boxes(
        [
            [200, 0, 260, 40],
            [0, 0, 100, 100],
            [0, 0, 100, 100],
            [400, 10, 460, 50],
        ],
        [0.50, 0.30, 0.80, 0.60],
        ["motorcycle", "car", "truck", "bus"],
    )
    kept = cw.select_vehicles(found)
    names = [str(name) for name in kept.data["class_name"]]
    confs = [round(float(conf), 5) for conf in kept.confidence]
    assert sorted(zip(confs, names)) == [
        (0.5, "motorcycle"),
        (0.6, "bus"),
        (0.8, "truck"),
    ]


def test_mean_below_20_is_dark_and_mean_20_is_counted(passthrough, tmp_path):
    low = np.full((16, 16, 3), 19, np.uint8)
    low.reshape(-1)[:100] = 20
    assert 19 < float(low.mean()) < 20
    even = np.full((16, 16, 3), 20, np.uint8)
    assert float(even.mean()) == 20.0
    model = RecordingModel(boxes([[1, 1, 8, 8]], [0.9], ["bus"]))
    snap = tmp_path / "s"
    snap.mkdir()
    dark_row = cw.read_camera(
        "alta-mp12.16", 137940, NOW, NOW_LABEL, model, str(snap),
        fetch=lambda url: low,
    )
    assert dark_row == {
        "ts": NOW_TS,
        "camera": "alta-mp12.16",
        "camera_id": 137940,
        "vehicles": None,
        "status": "dark",
    }
    assert model.calls == []
    assert list(snap.iterdir()) == []
    ok_row = cw.read_camera(
        "alta-mp12.16", 137940, NOW, NOW_LABEL, model, str(snap),
        fetch=lambda url: even,
    )
    assert ok_row["vehicles"] == 1
    assert type(ok_row["vehicles"]) is int
    assert ok_row["status"] == "ok"
    assert ok_row["ts"] == NOW_TS
    assert model.calls == [(0.25, (16, 16, 3))]
    assert (snap / "alta-mp12.16.jpg").is_file()


def test_infer_once_per_bright_frame_at_confidence_025(passthrough, tmp_path):
    model = RecordingModel(boxes([[4, 4, 12, 12]], [0.95], ["car"]))
    snap = tmp_path / "s"
    snap.mkdir()
    first = cw.read_camera(
        "white-pine-mp8.7", 142232, NOW, NOW_LABEL, model, str(snap),
        fetch=lambda url: bright(shape=(32, 48, 3)),
    )
    assert model.calls == [(0.25, (32, 48, 3))]
    second = cw.read_camera(
        "sr209-intersection", 141829, NOW, NOW_LABEL, model, str(snap),
        fetch=lambda url: bright(shape=(15, 18, 3)),
    )
    assert model.calls == [(0.25, (32, 48, 3)), (0.25, (15, 18, 3))]
    assert first["vehicles"] == 1
    assert second["vehicles"] == 1
    assert first["status"] == "ok"
    assert second["status"] == "ok"


def test_count_equals_boxes_after_nms(passthrough, tmp_path):
    # Truck duplicates the car box at a lower score, so NMS leaves 3 vehicles.
    found = boxes(
        [
            [0, 0, 30, 30],
            [50, 0, 80, 20],
            [0, 0, 30, 30],
            [100, 0, 130, 25],
        ],
        [0.9, 0.8, 0.4, 0.7],
        ["car", "bus", "truck", "motorcycle"],
    )
    assert len(cw.select_vehicles(found)) == 3
    snap = tmp_path / "s"
    snap.mkdir()
    reading = cw.read_camera(
        "seven-turns-mp7.4", 136332, NOW, NOW_LABEL, RecordingModel(found), str(snap),
        fetch=lambda url: bright(),
    )
    assert reading["vehicles"] == 3
    assert type(reading["vehicles"]) is int
    assert reading["status"] == "ok"


def test_infer_and_fetch_exceptions_become_error_rows(tmp_path):
    snap = tmp_path / "s"
    snap.mkdir()
    boom = Boom()
    infer_row = cw.read_camera(
        "white-pine-mp8.7", 142232, NOW, NOW_LABEL, boom, str(snap),
        fetch=lambda url: bright(),
    )
    assert boom.calls == [0.25]
    assert infer_row == {
        "ts": NOW_TS,
        "camera": "white-pine-mp8.7",
        "camera_id": 142232,
        "vehicles": None,
        "status": "error: detector failed",
    }
    assert list(snap.iterdir()) == []

    quiet = Boom()

    def fetch(url):
        raise TimeoutError("timed out")

    fetch_row = cw.read_camera(
        "alta-mp12.16", 137940, NOW, NOW_LABEL, quiet, str(snap),
        fetch=fetch,
    )
    assert quiet.calls == []
    assert fetch_row == {
        "ts": NOW_TS,
        "camera": "alta-mp12.16",
        "camera_id": 137940,
        "vehicles": None,
        "status": "error: timed out",
    }
    assert list(snap.iterdir()) == []


def test_canyon_total_sums_non_null_counts_as_int():
    total = cw.canyon_total(
        [
            {"vehicles": 2},
            {"vehicles": None},
            {"vehicles": 0},
            {"vehicles": 5},
        ],
        NOW,
        n_cams=4,
    )
    assert total == {
        "ts": NOW_TS,
        "camera": "_canyon_total",
        "vehicles": 7,
        "status": "3/4 cams ok",
    }
    assert type(total["vehicles"]) is int


def test_run_stamps_now_and_appends_the_next_poll(passthrough, tmp_path):
    first = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
    second = datetime(2026, 2, 1, 0, 10, 0, tzinfo=timezone.utc)
    model = RecordingModel(boxes([[2, 2, 12, 12]], [0.66], ["car"]))
    cameras = {"upper-vault-mp5.96": 142227}
    cw.run(model, str(tmp_path), now=first, fetch=lambda url: bright(), cameras=cameras)
    path = tmp_path / "readings.jsonl"
    after_first = path.read_text()
    cw.run(model, str(tmp_path), now=second, fetch=lambda url: bright(), cameras=cameras)
    after_second = path.read_text()
    assert after_second.startswith(after_first)
    assert len(after_second) > len(after_first)
    rows_first = [json.loads(line) for line in after_first.splitlines()]
    rows_all = [json.loads(line) for line in after_second.splitlines()]
    assert len(rows_first) == 2
    assert len(rows_all) == 4
    assert [row["ts"] for row in rows_first] == ["2026-02-01T00:00:00+00:00"] * 2
    assert [row["ts"] for row in rows_all[2:]] == ["2026-02-01T00:10:00+00:00"] * 2
    assert (tmp_path / "snapshots" / "20260201T000000Z" / "upper-vault-mp5.96.jpg").is_file()
    assert (tmp_path / "snapshots" / "20260201T001000Z" / "upper-vault-mp5.96.jpg").is_file()


def test_run_visits_cameras_in_dict_order(passthrough, tmp_path):
    seen = []

    def fetch(url):
        seen.append(url)
        return bright()

    cameras = {
        "white-pine-mp8.7": 142232,
        "alta-mp12.16": 137940,
        "sr209-intersection": 141829,
    }
    model = RecordingModel(boxes([[1, 1, 9, 9]], [0.7], ["truck"]))
    readings = cw.run(
        model, str(tmp_path), now=NOW, fetch=fetch, cameras=cameras,
    )
    assert seen == [
        "https://www.udottraffic.utah.gov/map/Cctv/142232",
        "https://www.udottraffic.utah.gov/map/Cctv/137940",
        "https://www.udottraffic.utah.gov/map/Cctv/141829",
    ]
    assert [row["camera"] for row in readings] == list(cameras)
    assert [row["camera_id"] for row in readings] == [142232, 137940, 141829]
    assert [row["vehicles"] for row in readings] == [1, 1, 1]
    assert [row["status"] for row in readings] == ["ok", "ok", "ok"]


def test_run_without_now_uses_aware_utc(passthrough, tmp_path, monkeypatch):
    fixed = datetime(2026, 10, 5, 16, 7, 8, tzinfo=timezone.utc)
    seen = []

    class Clock:
        @staticmethod
        def now(tz=None):
            seen.append(tz)
            return fixed

    monkeypatch.setattr(cw, "datetime", Clock)
    readings = cw.run(
        RecordingModel(boxes([[1, 1, 6, 6]], [0.7], ["truck"])),
        str(tmp_path),
        fetch=lambda url: bright(),
        cameras={"alta-bypass-mp10.95": 142061},
    )
    assert seen == [timezone.utc]
    assert readings[0]["status"] == "ok"
    assert readings[0]["vehicles"] == 1
    lines = (tmp_path / "readings.jsonl").read_text().splitlines()
    parsed = [datetime.fromisoformat(json.loads(line)["ts"]) for line in lines]
    assert parsed == [fixed, fixed]
    assert parsed[0].tzinfo is not None
    assert parsed[0].utcoffset() == timedelta(0)
    assert (tmp_path / "snapshots" / "20261005T160708Z" / "alta-bypass-mp10.95.jpg").is_file()


def test_jsonl_lines_match_consumer_keys(passthrough, tmp_path):
    found = boxes(
        [[0, 0, 10, 10], [20, 0, 30, 10]],
        [0.9, 0.8],
        ["car", "truck"],
    )
    dark = np.zeros((8, 8, 3), np.uint8)
    urls = {
        "https://www.udottraffic.utah.gov/map/Cctv/142232": bright(),
        "https://www.udottraffic.utah.gov/map/Cctv/137940": dark,
        "https://www.udottraffic.utah.gov/map/Cctv/141829": TimeoutError("timed out"),
        "https://www.udottraffic.utah.gov/map/Cctv/136332": bright(),
    }

    def fetch(url):
        payload = urls[url]
        if isinstance(payload, BaseException):
            raise payload
        return payload

    cameras = {
        "white-pine-mp8.7": 142232,
        "alta-mp12.16": 137940,
        "sr209-intersection": 141829,
        "seven-turns-mp7.4": 136332,
    }
    now = datetime(2026, 5, 6, 7, 8, 9, tzinfo=timezone.utc)
    cw.run(
        RecordingModel(found), str(tmp_path), now=now, fetch=fetch, cameras=cameras,
    )
    lines = (tmp_path / "readings.jsonl").read_text().splitlines()
    assert len(lines) == 5
    rows = [json.loads(line) for line in lines]
    for row in rows[:4]:
        assert list(row) == PER_CAM_KEYS
    assert list(rows[4]) == TOTAL_KEYS
    assert [row["ts"] for row in rows] == ["2026-05-06T07:08:09+00:00"] * 5
    assert [row["vehicles"] for row in rows[:4]] == [2, None, None, 2]
    assert [row["status"] for row in rows[:4]] == [
        "ok", "dark", "error: timed out", "ok",
    ]
    assert rows[4] == {
        "ts": "2026-05-06T07:08:09+00:00",
        "camera": "_canyon_total",
        "vehicles": 4,
        "status": "2/4 cams ok",
    }
    assert type(rows[4]["vehicles"]) is int
    snap = tmp_path / "snapshots" / "20260506T070809Z"
    assert sorted(path.name for path in snap.iterdir()) == [
        "seven-turns-mp7.4.jpg",
        "white-pine-mp8.7.jpg",
    ]


def test_snapshot_jpeg_has_input_shape_and_box(passthrough, tmp_path):
    # Text is drawn at y=700, so a 120px frame leaves the box as the annotation.
    img = np.full((120, 160, 3), 180, np.uint8)
    source = img.copy()
    x1, y1, x2, y2 = 36, 28, 110, 80
    found = boxes([[x1, y1, x2, y2]], [0.95], ["car"])
    snap = tmp_path / "s"
    snap.mkdir()
    reading = cw.read_camera(
        "white-pine-mp8.7", 142232, NOW, NOW_LABEL, RecordingModel(found), str(snap),
        fetch=lambda url: img,
    )
    assert reading["status"] == "ok"
    assert reading["vehicles"] == 1
    path = snap / "white-pine-mp8.7.jpg"
    raw = path.read_bytes()
    assert raw.startswith(b"\xff\xd8")
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == source.shape
    diff = np.abs(decoded.astype(np.int16) - source.astype(np.int16))
    assert int(diff[:8, :8].max()) <= 2
    region = diff[y1 - 2:y2 + 2, x1 - 2:x2 + 2]
    assert int(region.max()) >= 15


def test_summary_line_formats_zero_and_null_totals(passthrough, tmp_path, capsys):
    zero_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    null_at = datetime(2026, 1, 2, 4, 5, 6, tzinfo=timezone.utc)
    model = RecordingModel(empty_detections())
    cw.run(
        model,
        str(tmp_path),
        now=zero_at,
        fetch=lambda url: bright(),
        cameras={"sr209-intersection": 141829},
    )

    def fail_fetch(url):
        raise TimeoutError("down")

    cw.run(
        model,
        str(tmp_path),
        now=null_at,
        fetch=fail_fetch,
        cameras={"white-pine-mp8.7": 142232, "alta-mp12.16": 137940},
    )
    assert capsys.readouterr().out == (
        "20260102T030405Z total=0 sr209-intersection=0\n"
        "20260102T040506Z total=None white-pine-mp8.7=None alta-mp12.16=None\n"
    )
