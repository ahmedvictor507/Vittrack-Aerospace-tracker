import cv2
import numpy as np
import time

class AerospaceTracker:
    def __init__(self):
        self.patch_w, self.patch_h = 60, 60
        self.search_margin = 60
        self.target_hist = None
        self.target_template = None
        self.last_known_pos = None
        
        # Lucas-Kanade Optical Flow parameters
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_gray = None
        self.prev_pts = None

        # 6-State Kalman Filter: [x, y, vx, vy, ax, ay]
        self.kf = cv2.KalmanFilter(6, 2, 0)
        dt = 1.0
        # State Transition Matrix (Kinematic Equations with Acceleration)
        self.kf.transitionMatrix = np.array([
            [1, 0, dt,  0, 0.5*dt**2,         0],
            [0, 1,  0, dt,         0, 0.5*dt**2],
            [0, 0,  1,  0,        dt,         0],
            [0, 0,  0,  1,         0,        dt],
            [0, 0,  0,  0,         1,         0],
            [0, 0,  0,  0,         0,         1]
        ], dtype=np.float32)
        
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.05
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        self.frames_lost = 0
        self.max_lost_frames = 30

    def init_target(self, frame, pt):
        px, py = pt
        h, w = frame.shape[:2]
        
        xmin, ymin = max(0, px - self.patch_w // 2), max(0, py - self.patch_h // 2)
        xmax, ymax = min(w, px + self.patch_w // 2), min(h, py + self.patch_h // 2)
        crop = frame[ymin:ymax, xmin:xmax]
        if crop.size == 0: return False

        # Optimized 32x32 Histogram for Edge processing
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        self.target_hist = cv2.calcHist([hsv_crop], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(self.target_hist, self.target_hist, 0, 255, cv2.NORM_MINMAX)
        
        self.target_template = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_pts = np.array([[[float(px), float(py)]]], dtype=np.float32)
        
        self.kf.statePost = np.array([[px], [py], [0], [0], [0], [0]], dtype=np.float32)
        self.last_known_pos = (px, py)
        self.frames_lost = 0
        return True

    def track(self, frame):
        if self.target_template is None: return None, 0.0
        
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 1. Optical Flow Prediction (Micro-movement)
        flow_cx, flow_cy = self.last_known_pos
        if self.prev_gray is not None and self.prev_pts is not None:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, **self.lk_params)
            if status[0][0] == 1:
                flow_cx, flow_cy = next_pts[0][0].ravel()
        
        # 2. Kalman Prediction (Macro-movement)
        pred = self.kf.predict()
        vx, vy = pred[2][0], pred[3][0]
        
        # Dynamic Search Margin based on velocity
        dynamic_margin = max(60, int(np.sqrt(vx**2 + vy**2) * 1.5) + 30)
        
        if self.frames_lost > self.max_lost_frames:
            # Global Re-detection Mode
            roi_xmin, roi_ymin, roi_xmax, roi_ymax = 0, 0, w, h
        else:
            # Localized Search
            search_cx = int((flow_cx + pred[0][0]) / 2)
            search_cy = int((flow_cy + pred[1][0]) / 2)
            roi_xmin = max(0, search_cx - dynamic_margin)
            roi_ymin = max(0, search_cy - dynamic_margin)
            roi_xmax = min(w, search_cx + dynamic_margin)
            roi_ymax = min(h, search_cy + dynamic_margin)

        roi_frame = frame[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
        if roi_frame.shape[0] < self.patch_h or roi_frame.shape[1] < self.patch_w:
            self.frames_lost += 1
            return self.last_known_pos, 0.0

        # 3. Robust Signal Fusion (Weighted, NOT bitwise)
        roi_hsv = hsv[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
        roi_gray = gray[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
        
        color_map = cv2.calcBackProject([roi_hsv], [0, 1], self.target_hist, [0, 180, 0, 256], 1)
        edge_map = cv2.matchTemplate(roi_gray, self.target_template, cv2.TM_CCOEFF_NORMED)
        edge_map = cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Pad edge map to match color map size
        pad_y, pad_x = self.patch_h // 2, self.patch_w // 2
        edge_map_padded = cv2.copyMakeBorder(edge_map, pad_y, self.patch_h - pad_y - 1, 
                                             pad_x, self.patch_w - pad_x - 1, cv2.BORDER_CONSTANT, value=0)
        
        # Weighted Fusion: 60% Structure, 40% Color
        fused = cv2.addWeighted(edge_map_padded, 0.6, color_map, 0.4, 0)
        _, max_val, _, max_loc = cv2.minMaxLoc(fused)
        confidence = max_val / 255.0

        if confidence > 0.4:
            # Target acquired
            raw_cx = max_loc[0] + roi_xmin
            raw_cy = max_loc[1] + roi_ymin
            
            # Kalman Correction
            self.kf.correct(np.array([[np.float32(raw_cx)], [np.float32(raw_cy)]]))
            self.last_known_pos = (raw_cx, raw_cy)
            self.prev_pts = np.array([[[float(raw_cx), float(raw_cy)]]], dtype=np.float32)
            self.frames_lost = 0
            
            # 4. Template Appearance EMA Update
            if confidence > 0.75:
                tx_min = max(0, raw_cx - self.patch_w // 2)
                ty_min = max(0, raw_cy - self.patch_h // 2)
                tx_max = min(w, tx_min + self.patch_w)
                ty_max = min(h, ty_min + self.patch_h)
                
                current_patch = gray[ty_min:ty_max, tx_min:tx_max]
                if current_patch.shape == self.target_template.shape:
                    self.target_template = cv2.addWeighted(self.target_template, 0.95, current_patch, 0.05, 0)
        else:
            # Signal Lost: Coast on physics and flow
            self.last_known_pos = (int(pred[0][0]), int(pred[1][0]))
            self.frames_lost += 1

        self.prev_gray = gray.copy()
        return self.last_known_pos, confidence


clicked_pt = None
def mouse_cb(event, x, y, flags, param):
    global clicked_pt
    if event == cv2.EVENT_LBUTTONDOWN: clicked_pt = (x, y)

def main():
    global clicked_pt
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    # Critical for Jetson latency: Drop stale frames in the hardware buffer
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    tracker = AerospaceTracker()
    cv2.namedWindow("Flight Tracker")
    cv2.setMouseCallback("Flight Tracker", mouse_cb)

    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        
        if clicked_pt:
            tracker.init_target(frame, clicked_pt)
            clicked_pt = None
            
        pos, conf = tracker.track(frame)
        
        if pos and tracker.target_template is not None:
            cx, cy = pos
            color = (0, 255, 0) if conf > 0.4 else (0, 0, 255)
            if tracker.frames_lost > tracker.max_lost_frames:
                cv2.putText(frame, "RE-DETECTING (GLOBAL)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                color = (0, 165, 255)
                
            pw, ph = tracker.patch_w // 2, tracker.patch_h // 2
            cv2.rectangle(frame, (cx - pw, cy - ph), (cx + pw, cy + ph), color, 2)
            cv2.putText(frame, f"Conf: {conf:.2f} | Lost: {tracker.frames_lost}", (cx - pw, cy - ph - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if conf > 0.4 and tracker.frames_lost <= tracker.max_lost_frames:
                # Check Attack Mode
                import os
                attack_mode = False
                try:
                    if os.path.exists("/tmp/drone_attack_state"):
                        with open("/tmp/drone_attack_state", "r") as f:
                            if f.read().strip() == "1":
                                attack_mode = True
                except:
                    pass

                if attack_mode:
                    frame_h, frame_w = frame.shape[:2]
                    center_x, center_y = frame_w // 2, frame_h // 2
                    
                    err_x = cx - center_x
                    err_y = cy - center_y
                    
                    if abs(err_x) < 15: err_x = 0
                    if abs(err_y) < 15: err_y = 0
                    
                    if err_x != 0 or err_y != 0:
                        print(f"[MAVLink SIM] ATTACKING! dx:{err_x}, dy:{err_y}")
                        cv2.putText(frame, f"ATTACKING! dx:{err_x} dy:{err_y}", (20, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        print("[MAVLink SIM] ATTACK LOCKED - HOLDING POSITION")
                        cv2.putText(frame, "ATTACK LOCKED - DEAD CENTER", (20, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
        cv2.imshow("Flight Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()