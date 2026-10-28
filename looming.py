#!/usr/bin/env python3
"""
Fly-style looming + optic-flow detector for Raspberry Pi 4 + Camera Module 3.

Prints two signals every frame:
  - expansion: how strongly the central image is spreading outward
               (an approaching object -> positive; the LPLC2-style looming cue)
  - L/R flow:  average motion on the left vs right half of the image
               (the cue flies and bees use to centre themselves in corridors)

Run:  python3 looming.py   (Ctrl+C to stop)
"""
import time
import numpy as np
import cv2
from picamera2 import Picamera2

W, H = 160, 120     # deliberately low resolution, like a fly's coarse eye
THRESH = 0.02       # expansion level that counts as "looming" - tune by watching the numbers
SMOOTH = 0.3        # 0..1, higher = faster response but noisier

cam = Picamera2()
cam.configure(cam.create_video_configuration(main={"size": (W, H), "format": "RGB888"}))
cam.start()
time.sleep(1)  # let auto-exposure settle

prev = cv2.cvtColor(cam.capture_array(), cv2.COLOR_BGR2GRAY)
expansion = 0.0
t_last = time.time()

try:
    while True:
        frame = cv2.cvtColor(cam.capture_array(), cv2.COLOR_BGR2GRAY)

        # Dense optic flow: for every pixel, where did it move since the last frame?
        flow = cv2.calcOpticalFlowFarneback(prev, frame, None, 0.5, 2, 9, 2, 5, 1.1, 0)
        u, v = flow[..., 0], flow[..., 1]

        # Divergence = how much the flow field points outward.
        # Something growing in the centre of view -> pixels move away from the centre -> positive.
        div = np.gradient(u, axis=1) + np.gradient(v, axis=0)
        cy, cx = H // 2, W // 2
        centre = div[cy - 30:cy + 30, cx - 40:cx + 40].mean()
        expansion = (1 - SMOOTH) * expansion + SMOOTH * centre

        # Left/right flow balance (corridor centring)
        mag = np.hypot(u, v)
        left, right = mag[:, :W // 2].mean(), mag[:, W // 2:].mean()

        now = time.time()
        fps = 1.0 / max(now - t_last, 1e-6)
        t_last = now

        flag = "  << LOOMING!" if expansion > THRESH else ""
        print(f"{fps:5.1f} fps | expansion {expansion:+.3f} | L/R flow {left:.2f}/{right:.2f}{flag}")

        prev = frame
except KeyboardInterrupt:
    pass
finally:
    cam.stop()
