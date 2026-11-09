"""Collect training data.

Hold up a number of fingers and press that number (0-5) on the keyboard.
Each press saves:
  - the camera photo   -> data/images/<number>/<name>.jpg
  - one row of data    -> data/labels.csv (label + all 21 joint positions)

Keys:
  0-5   save a sample with that label (hold the key down for a quick burst)
  d     turn the debug overlay on/off
  q/Esc quit

The window must be focused (clicked) for key presses to register.
"""

import csv
import time
from datetime import datetime

import cv2

from common import (
    GREEN, IMAGES_DIR, LABELS_CSV, NUM_CLASSES, NUM_LANDMARKS, RED, WHITE, WINDOW_NAME,
    FPSCounter, HandTracker, draw_hand_debug, draw_text, camera_index_from_args, open_camera, read_mirrored,
)

CSV_HEADER = (
    ["filename", "label", "handedness", "hand_score", "img_w", "img_h", "timestamp"]
    + [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in "xyz"]
)
FLASH_SECONDS = 0.25


def load_counts() -> list[int]:
    """How many samples already exist per label (so the counter survives restarts)."""
    counts = [0] * NUM_CLASSES
    if LABELS_CSV.exists():
        with open(LABELS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                counts[int(row["label"])] += 1
    return counts


def save_sample(frame, hand, label: int, csv_writer):
    """Write the photo to its label's folder and append a row to labels.csv."""
    now = datetime.now()
    name = now.strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
    rel_path = f"images/{label}/{name}"
    (IMAGES_DIR / str(label)).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(IMAGES_DIR / str(label) / name), frame)

    h, w = frame.shape[:2]
    row = [rel_path, label, hand.handedness, f"{hand.score:.4f}", w, h, now.isoformat()]
    row += [f"{v:.6f}" for v in hand.landmarks.reshape(-1)]
    csv_writer.writerow(row)


def main():
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not LABELS_CSV.exists()
    csv_file = open(LABELS_CSV, "a", newline="")
    writer = csv.writer(csv_file)
    if is_new_file:
        writer.writerow(CSV_HEADER)

    counts = load_counts()
    cap = open_camera(camera_index_from_args())
    tracker = HandTracker()
    fps = FPSCounter()
    debug = False
    flash_until, flash_color, flash_text = 0.0, GREEN, ""

    try:
        while True:
            frame = read_mirrored(cap)
            if frame is None:
                continue
            hand = tracker.process(frame)
            clean = frame.copy()           # the photo we save, without any overlay
            current_fps = fps.tick()

            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):      # 27 = Esc
                break
            if key == ord("d"):
                debug = not debug
            if ord("0") <= key <= ord(str(NUM_CLASSES - 1)):
                label = key - ord("0")
                if hand is None:
                    flash_color, flash_text = RED, "No hand detected - not saved"
                else:
                    save_sample(clean, hand, label, writer)
                    csv_file.flush()       # write to disk now, in case of a crash
                    counts[label] += 1
                    flash_color, flash_text = GREEN, f"Saved {label}  (#{counts[label]})"
                flash_until = time.monotonic() + FLASH_SECONDS

            # Overlay
            if debug:
                if hand is not None:
                    draw_hand_debug(frame, hand)
                draw_text(frame, f"FPS {current_fps:4.1f}", (10, 25))
                draw_text(frame, "hand: yes" if hand else "hand: NO", (10, 50),
                          color=GREEN if hand else RED)

            # Always-visible: per-label sample counts along the bottom
            h, w = frame.shape[:2]
            counter = "   ".join(f"{i}:{c}" for i, c in enumerate(counts))
            draw_text(frame, counter, (10, h - 15), 0.6)
            draw_text(frame, "0-5 save | d debug | q quit", (w - 245, 25), 0.45)

            # Brief colored border + message after each key press
            if time.monotonic() < flash_until:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), flash_color, 8)
                draw_text(frame, flash_text, (10, h - 45), 0.7, WHITE, 2, bg=flash_color)

            cv2.imshow(WINDOW_NAME, frame)
    finally:
        csv_file.close()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Samples per label:", dict(enumerate(counts)))


if __name__ == "__main__":
    main()
