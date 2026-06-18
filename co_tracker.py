"""
Real-Time Webcam Tracking Pipeline using CoTracker (Meta AI)
============================================================
Controls:
  LEFT CLICK  – add a tracking point at cursor position
  R           – reset all tracked points
  G           – auto-place a 5×5 grid of tracking points
  Q / ESC     – quit

Installation (run once):
  pip install torch torchvision opencv-python einops timm
  pip install git+https://github.com/facebookresearch/co-tracker.git
"""

import sys
import time
import collections
from typing import Optional

import cv2
import numpy as np
import torch

# ── CoTracker import ────────────────────────────────────────────────────────
try:
    from cotracker.predictor import CoTrackerOnlinePredictor
except ImportError:
    sys.exit(
        "[ERROR] cotracker not found.\n"
        "Install with:\n"
        "  pip install git+https://github.com/facebookresearch/co-tracker.git"
    )


# ── Config ───────────────────────────────────────────────────────────────────
WINDOW_FRAMES   = 8          # CoTracker online window length (must be ≥ 4)
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
TRAIL_LENGTH    = 30         # how many past positions to draw as a trail
POINT_RADIUS    = 6
TRAIL_THICKNESS = 2
FONT            = cv2.FONT_HERSHEY_SIMPLEX


# ── Colour palette (BGR) ─────────────────────────────────────────────────────
PALETTE = [
    (0, 255, 127), (0, 165, 255), (255, 50,  50),  (255, 255,  0),
    (180, 0, 255), (0, 255, 255), (255, 128,  0),  (128, 255,  0),
    (255,  0, 180),(0, 200, 255),
]


def colour_for(idx: int):
    return PALETTE[idx % len(PALETTE)]


# ── Model loader ─────────────────────────────────────────────────────────────
def load_model() -> CoTrackerOnlinePredictor:
    print(f"[INFO] Loading CoTracker (online) on {DEVICE} …")
    model = CoTrackerOnlinePredictor(checkpoint=None)   # downloads weights automatically
    model = model.to(DEVICE)
    model.eval()
    print("[INFO] Model ready.")
    return model


# ── Frame → tensor ───────────────────────────────────────────────────────────
def frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    """BGR uint8 HWC  →  float32 1×1×3×H×W in [0,1]"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t   = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0   # 3×H×W
    return t.unsqueeze(0).unsqueeze(0).to(DEVICE)                   # 1×1×3×H×W


# ── Mouse callback ───────────────────────────────────────────────────────────
class ClickHandler:
    def __init__(self):
        self.new_points: list[tuple[int, int]] = []

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.new_points.append((x, y))
            print(f"[CLICK] Point queued at ({x}, {y})")


# ── Tracker state ─────────────────────────────────────────────────────────────
class TrackerState:
    def __init__(self, model: CoTrackerOnlinePredictor):
        self.model   = model
        self.queries : Optional[torch.Tensor] = None   # N×3 [t, x, y]
        self.trails  : dict[int, collections.deque]  = {}
        self.visible : dict[int, bool]               = {}
        self.frame_idx = 0
        self._reset_model()

    def _reset_model(self):
        self.model.reset()                 # clear internal hidden state
        self.frame_idx = 0
        self.queries   = None
        self.trails    = {}
        self.visible   = {}

    def reset(self):
        print("[INFO] Tracker reset.")
        self._reset_model()

    def add_points(self, xy_list: list[tuple[int, int]]):
        """Append new query points stamped at the current frame."""
        if not xy_list:
            return
        new_q = torch.tensor(
            [[self.frame_idx, float(x), float(y)] for x, y in xy_list],
            dtype=torch.float32, device=DEVICE
        )
        if self.queries is None:
            self.queries = new_q
        else:
            self.queries = torch.cat([self.queries, new_q], dim=0)

        # initialise trail storage
        start = 0 if self.queries is None else len(self.queries) - len(xy_list)
        for i in range(start, len(self.queries)):
            self.trails[i]  = collections.deque(maxlen=TRAIL_LENGTH)
            self.visible[i] = True

        print(f"[INFO] Tracking {len(self.queries)} point(s).")

    def add_grid(self, w: int, h: int, rows: int = 5, cols: int = 5):
        pts = [
            (int(w * (c + 0.5) / cols), int(h * (r + 0.5) / rows))
            for r in range(rows) for c in range(cols)
        ]
        self.add_points(pts)

    def step(self, frame_tensor: torch.Tensor) -> Optional[np.ndarray]:
        """
        Feed one frame to the model.  Returns predicted (x,y,vis) per track,
        or None if there are no queries yet.
        """
        if self.queries is None or len(self.queries) == 0:
            self.frame_idx += 1
            return None

        # CoTrackerOnlinePredictor.step expects:
        #   frame : 1×1×3×H×W
        #   queries : 1×N×3  (optional after first call)
        q = self.queries.unsqueeze(0)          # 1×N×3
        with torch.no_grad():
            pred_tracks, pred_vis = self.model(
                frame_tensor,
                queries=q,
                add_support_grid=True,
            )
        # pred_tracks : 1×T×N×2,  pred_vis : 1×T×N  (T = window length)
        # We only care about the LAST time step in the window
        tracks = pred_tracks[0, -1].cpu().numpy()   # N×2
        vis    = pred_vis[0,   -1].cpu().numpy()    # N   (bool)

        for i in range(len(tracks)):
            xy = (int(tracks[i, 0]), int(tracks[i, 1]))
            self.trails[i].append(xy)
            self.visible[i] = bool(vis[i])

        self.frame_idx += 1
        return tracks, vis.astype(bool)


# ── Overlay renderer ──────────────────────────────────────────────────────────
def draw_overlay(
    canvas: np.ndarray,
    state:  TrackerState,
    fps:    float,
    n_pts:  int,
):
    # Trails + current positions
    for i, trail in state.trails.items():
        col = colour_for(i)
        pts = list(trail)
        # draw trail
        for j in range(1, len(pts)):
            alpha = j / len(pts)
            c = tuple(int(v * alpha) for v in col)
            cv2.line(canvas, pts[j - 1], pts[j], c, TRAIL_THICKNESS)
        # draw current point
        if pts:
            is_vis = state.visible.get(i, True)
            cv2.circle(canvas, pts[-1], POINT_RADIUS, col, -1 if is_vis else 2)

    # HUD
    dev_str = DEVICE.upper()
    lines = [
        f"FPS: {fps:5.1f}",
        f"Points: {n_pts}",
        f"Device: {dev_str}",
        "L-click: add point",
        "G: grid  R: reset  Q: quit",
    ]
    for k, txt in enumerate(lines):
        y = 22 + k * 22
        cv2.putText(canvas, txt, (10, y), FONT, 0.55, (0, 0, 0),   2, cv2.LINE_AA)
        cv2.putText(canvas, txt, (10, y), FONT, 0.55, (220, 220, 220), 1, cv2.LINE_AA)


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    model   = load_model()
    state   = TrackerState(model)
    clicker = ClickHandler()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        sys.exit("[ERROR] Cannot open webcam (index 0).")

    cv2.namedWindow("CoTracker – Webcam", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("CoTracker – Webcam", clicker.callback)

    fps_timer = time.perf_counter()
    fps       = 0.0
    frame_buf: list[np.ndarray] = []

    print("[INFO] Webcam feed live. Press G for a grid of points, click to add custom points.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame grab failed – retrying…")
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = cv2.flip(frame, 1)        # mirror for natural feel

        # ── handle new click-points ──────────────────────────────────────────
        if clicker.new_points:
            state.add_points(clicker.new_points)
            clicker.new_points.clear()

        # ── run model ────────────────────────────────────────────────────────
        tensor = frame_to_tensor(frame)
        state.step(tensor)

        # ── draw ─────────────────────────────────────────────────────────────
        canvas = frame.copy()
        n_pts  = len(state.queries) if state.queries is not None else 0
        draw_overlay(canvas, state, fps, n_pts)

        cv2.imshow("CoTracker – Webcam", canvas)

        # ── FPS ──────────────────────────────────────────────────────────────
        now = time.perf_counter()
        fps = 1.0 / max(now - fps_timer, 1e-6)
        fps_timer = now

        # ── key handling ─────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):          # Q or ESC
            break
        elif key == ord("r"):
            state.reset()
        elif key == ord("g"):
            state.add_grid(FRAME_WIDTH, FRAME_HEIGHT)

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()