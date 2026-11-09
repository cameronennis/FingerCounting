"""Live test: shows how many fingers you're holding up, updating every frame.

Keys:
  d     turn the debug overlay on/off (joints, hand box, FPS, per-number confidence)
  q/Esc quit
"""

from collections import deque

import cv2
import numpy as np
import torch

from common import (
    CLASSIFIER_PATH, GREEN, NUM_CLASSES, RED, WHITE, WINDOW_NAME,
    FingerMLP, FPSCounter, HandTracker, draw_hand_debug, draw_text,
    normalize_landmarks, camera_index_from_args, open_camera, read_mirrored,
)

SMOOTHING_FRAMES = 5   # average the last few predictions so the number doesn't flicker


def load_classifier() -> FingerMLP:
    if not CLASSIFIER_PATH.exists():
        raise SystemExit(f"No trained model at {CLASSIFIER_PATH}. Run train.py first.")
    model = FingerMLP()
    model.load_state_dict(torch.load(CLASSIFIER_PATH, weights_only=True))
    model.eval()                                   # dropout off for predictions
    return model


@torch.no_grad()
def predict_probs(model, hand, img_w, img_h) -> np.ndarray:
    """Return 6 probabilities (one per finger count) that add up to 1."""
    features = normalize_landmarks(hand.landmarks, hand.handedness, img_w, img_h)
    scores = model(torch.from_numpy(features).unsqueeze(0))
    return torch.softmax(scores, dim=1)[0].numpy()  # softmax turns scores into probabilities


def draw_prob_bars(frame, probs):
    """Debug: one horizontal bar per finger count showing the model's confidence."""
    x, y0, bar_w, bar_h = 10, 80, 150, 16
    for i, p in enumerate(probs):
        y = y0 + i * (bar_h + 6)
        color = GREEN if i == probs.argmax() else WHITE
        cv2.rectangle(frame, (x + 22, y), (x + 22 + bar_w, y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (x + 22, y), (x + 22 + int(bar_w * p), y + bar_h), color, -1)
        cv2.putText(frame, str(i), (x, y + bar_h - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{p:.0%}", (x + 28 + bar_w, y + bar_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def main():
    model = load_classifier()
    cap = open_camera(camera_index_from_args())
    tracker = HandTracker()
    fps = FPSCounter()
    recent = deque(maxlen=SMOOTHING_FRAMES)
    debug = False

    try:
        while True:
            frame = read_mirrored(cap)
            if frame is None:
                continue
            h, w = frame.shape[:2]
            hand = tracker.process(frame)
            current_fps = fps.tick()

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                debug = not debug

            if hand is None:
                recent.clear()                     # start fresh when the hand returns
                probs = None
            else:
                recent.append(predict_probs(model, hand, w, h))
                probs = np.mean(recent, axis=0)

            # Big finger count, always shown
            if probs is not None:
                count = int(probs.argmax())
                cv2.putText(frame, str(count), (w - 110, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            4, (0, 0, 0), 14, cv2.LINE_AA)   # dark outline
                cv2.putText(frame, str(count), (w - 110, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            4, GREEN, 6, cv2.LINE_AA)
            else:
                draw_text(frame, "No hand", (w - 120, 40), 0.7, RED, 2)

            if debug:
                if hand is not None:
                    draw_hand_debug(frame, hand)
                draw_text(frame, f"FPS {current_fps:4.1f}", (10, 25))
                if probs is not None:
                    draw_text(frame, f"confidence {probs.max():.0%}", (10, 55))
                    draw_prob_bars(frame, probs)

            draw_text(frame, "d debug | q quit", (10, h - 15), 0.45, WHITE)
            cv2.imshow(WINDOW_NAME, frame)
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
