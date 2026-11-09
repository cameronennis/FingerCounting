# Finger Counting Model

Counts how many fingers (0–5) you're holding up, live from your webcam. Google's MediaPipe finds 21 hand joint positions, and a small PyTorch network trained on your own samples turns them into a count.

Tested on macOS. It should also work on Windows and Linux.

## Requirements

- Python 3.14 (the versions in `requirements.txt` were tested on it)
- A webcam/connected camera (e.g. iPhone)

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

On Windows, use `.venv\Scripts\pip` and `.venv\Scripts\python` instead.

The first run downloads Google's hand model (~8 MB) into `models/`.

## Usage

1. **Collect:** `.venv/bin/python collect.py`. Hold up fingers and press the matching key (`0`–`5`) to save a sample. Aim for 50+ per number, with varied angles and distances.
2. **Train:** `.venv/bin/python train.py`. Prints test accuracy and a confusion table.
3. **Test live:** `.venv/bin/python live.py`.

Re-run `train.py` after every collection session. New samples aren't used until you do.

**Keys:** `d` toggles the debug overlay (joints, hand box, FPS, confidence). `q` or `Esc` quits. Click the video window first so it receives key presses.

**Choosing a camera:** add `--camera N` to `collect.py` or `live.py`, e.g. `--camera 1`.

## macOS notes

- **Camera permission:** the app running the script (Terminal, VS Code, ...) needs camera access in System Settings → Privacy & Security → Camera. Restart that app after allowing it.
- **Black screen:** an iPhone connected through Continuity Camera can take camera 0 and sends a black picture while locked. Try `--camera 1`, or turn it off on the iPhone: Settings → General → AirPlay & Continuity → Continuity Camera.
- **Balloons, thumbs-up bubbles, fireworks:** macOS Reactions trigger on hand gestures. Turn them off from the green camera icon in the menu bar → Reactions.
