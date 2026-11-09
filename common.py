"""Shared code used by collect.py, train.py and live.py.

Contents:
  - file paths and constants
  - opening the camera
  - HandTracker: runs Google's MediaPipe hand model on each camera frame
  - normalize_landmarks: turns 21 raw joint positions into model input
  - FingerMLP: the small neural network we train
  - drawing helpers for the on-screen overlay
"""

import argparse
import ssl
import sys
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import certifi
import cv2
import numpy as np
import torch
from torch import nn
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions, vision

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
LABELS_CSV = DATA_DIR / "labels.csv"
HAND_MODEL_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"
CLASSIFIER_PATH = PROJECT_DIR / "models" / "finger_mlp.pt"

NUM_CLASSES = 6        # 0, 1, 2, 3, 4, 5 fingers
NUM_LANDMARKS = 21     # MediaPipe finds 21 points per hand
FEATURES = NUM_LANDMARKS * 3  # x, y, z for each point = 63 numbers

WINDOW_NAME = "Finger Counter"

# Colors are (Blue, Green, Red) in OpenCV.
GREEN = (80, 220, 80)
RED = (60, 60, 230)
WHITE = (255, 255, 255)
YELLOW = (0, 220, 255)
CYAN = (230, 200, 40)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def camera_index_from_args() -> int:
    """Read the optional --camera N argument (default 0).

    Macs can have several cameras (built-in, iPhone via Continuity Camera, ...).
    Example: .venv/bin/python collect.py --camera 1
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="camera number: 0, 1, 2, ...")
    return parser.parse_args().camera


def open_camera(index: int = 0, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    """Open the webcam. 640x480 keeps things fast; MediaPipe doesn't need more."""
    # AVFoundation is the macOS camera system; elsewhere let OpenCV pick
    # (DirectShow/Media Foundation on Windows, V4L2 on Linux).
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {index}. On macOS, allow camera access for the app "
            "running this script (System Settings > Privacy & Security > Camera), "
            "then restart that app. If OpenCV said 'out device of bound (0-N)', "
            "only cameras 0 to N exist right now: pick one of those with --camera."
        )

    # Cameras often send a few dark frames while starting up. If the picture
    # is still black after ~2 seconds, this is probably the wrong camera
    # (e.g. a locked iPhone connected through Continuity Camera).
    for _ in range(60):
        ok, frame = cap.read()
        if ok and frame.mean() > 5:
            print(f"Using camera {index}.")
            break
    else:
        print(f"WARNING: camera {index} is only sending black frames. "
              f"Quit (q) and try --camera {index + 1}.")
    return cap


def read_mirrored(cap: cv2.VideoCapture) -> np.ndarray | None:
    """Grab one frame and flip it left-right so the screen behaves like a mirror."""
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.flip(frame, 1)


# ---------------------------------------------------------------------------
# Hand tracking (MediaPipe)
# ---------------------------------------------------------------------------

@dataclass
class Hand:
    """One detected hand.

    landmarks:  21 x 3 array. x and y are fractions of the image width/height
                (0 to 1). z is depth relative to the wrist, on about the same
                scale as x (smaller = closer to the camera).
    handedness: "Left" or "Right"
    score:      how sure MediaPipe is about the handedness (0 to 1)
    """
    landmarks: np.ndarray
    handedness: str
    score: float


HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def ensure_hand_model():
    """Download Google's hand model (~8 MB) on first run if it isn't there yet."""
    if HAND_MODEL_PATH.exists():
        return
    print("Downloading MediaPipe hand model (~8 MB, first run only)...")
    HAND_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = HAND_MODEL_PATH.with_suffix(".part")   # so a failed download never looks complete
    try:
        # certifi supplies up-to-date security certificates; some Python
        # installs (e.g. python.org on macOS) don't have any set up by default.
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(HAND_MODEL_URL, context=context) as response:
            partial.write_bytes(response.read())
    except Exception as e:
        partial.unlink(missing_ok=True)
        raise SystemExit(
            f"Could not download the hand model ({e}).\n"
            f"Download it manually from:\n  {HAND_MODEL_URL}\n"
            f"and save it as:\n  {HAND_MODEL_PATH}"
        )
    partial.rename(HAND_MODEL_PATH)
    print(f"Saved to {HAND_MODEL_PATH}")


class HandTracker:
    """Wraps MediaPipe's HandLandmarker in VIDEO mode.

    VIDEO mode is what makes it fast: once the hand is found, MediaPipe uses
    the previous frame's hand position as the crop for the next frame, and
    only reruns the (slower) palm detector when it loses the hand.
    """

    def __init__(self, min_confidence: float = 0.5):
        ensure_hand_model()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray) -> Hand | None:
        """Find the hand in one frame. Returns None if no hand is visible."""
        # VIDEO mode needs a timestamp that always goes up.
        ts = int(time.monotonic() * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        # OpenCV stores pixels as BGR; MediaPipe expects RGB.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(Image(image_format=ImageFormat.SRGB, data=rgb), ts)

        if not result.hand_landmarks:
            return None
        points = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
        category = result.handedness[0][0]
        # MediaPipe labels handedness as if the image were NOT mirrored. We
        # flip every frame to make a mirror view, so swap the label back.
        handedness = {"Left": "Right", "Right": "Left"}.get(category.category_name, category.category_name)
        return Hand(points, handedness, float(category.score))

    def close(self):
        self._landmarker.close()


# ---------------------------------------------------------------------------
# Turning joint positions into model input
# ---------------------------------------------------------------------------

def normalize_landmarks(landmarks: np.ndarray, handedness: str, img_w: int, img_h: int) -> np.ndarray:
    """Convert 21 raw joint positions into 63 numbers that describe hand *shape* only.

    Steps:
      1. Convert to pixel units, so a square hand stays square (x and y are
         otherwise fractions of different lengths: 640 wide vs 480 tall).
      2. Shift so the wrist (point 0) is at (0, 0, 0): where the hand is on
         screen no longer matters.
      3. Mirror left hands so every hand looks like a right hand: either hand works.
      4. Scale so the farthest point from the wrist is 1 unit away: how close
         the hand is to the camera no longer matters.
    """
    pts = landmarks.astype(np.float32).copy()
    pts[:, 0] *= img_w
    pts[:, 1] *= img_h
    pts[:, 2] *= img_w          # MediaPipe's z uses roughly the same scale as x

    pts -= pts[0]               # wrist to origin

    if handedness == "Left":
        pts[:, 0] *= -1         # mirror left to right

    size = np.linalg.norm(pts, axis=1).max()
    if size > 0:
        pts /= size

    return pts.reshape(-1)      # 21 x 3  ->  63


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------

class FingerMLP(nn.Module):
    """A small fully connected network: 63 numbers in -> 6 scores out.

    The highest score is the predicted finger count. Dropout randomly switches
    off some neurons during training, which helps the model not just memorize
    the training samples.
    """

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURES, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


def draw_text(frame, text, org, scale=0.6, color=WHITE, thickness=1, bg=(0, 0, 0)):
    """Text with a dark box behind it, so it's readable on any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), base = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - 4, y - h - 4), (x + w + 4, y + base + 2), bg, -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_hand_debug(frame, hand: Hand):
    """Debug overlay for a hand: joint points, bones, hand box, handedness + score."""
    h, w = frame.shape[:2]
    px = (hand.landmarks[:, :2] * [w, h]).astype(int)

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(px[a]), tuple(px[b]), WHITE, 2, cv2.LINE_AA)
    for x, y in px:
        cv2.circle(frame, (x, y), 4, CYAN, -1, cv2.LINE_AA)

    # Hand box: the joints' extent plus some padding. This closely matches the
    # area MediaPipe crops to internally when tracking.
    pad = 20
    x0, y0 = px.min(axis=0) - pad
    x1, y1 = px.max(axis=0) + pad
    cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), YELLOW, 2)
    draw_text(frame, f"{hand.handedness} {hand.score:.2f}", (int(x0), int(y0) - 8), 0.5, YELLOW)


class FPSCounter:
    """Frames per second, averaged over the last 30 frames."""

    def __init__(self):
        self.times = deque(maxlen=30)

    def tick(self) -> float:
        self.times.append(time.perf_counter())
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / (self.times[-1] - self.times[0])
