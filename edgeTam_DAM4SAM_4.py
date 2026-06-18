import cv2
import numpy as np
import time

class DAM4SAM_MemoryManager:
    """
    Suppresses background environmental noise and lookalike distractors
    by keeping track of localized background profiles.
    """
    def __init__(self):
        self.background_hist = None
        self.alpha = 0.15  

    def update_background(self, frame_hsv, target_roi_box):
        x, y, w, h = target_roi_box
        mask = np.ones(frame_hsv.shape[:2], dtype=np.uint8) * 255
        mask[max(0, y):min(mask.shape[0], y+h), max(0, x):min(mask.shape[1], x+w)] = 0
        
        hist = cv2.calcHist([frame_hsv], [0, 1], mask, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        
        if self.background_hist is None:
            self.background_hist = hist
        else:
            self.background_hist = cv2.addWeighted(self.background_hist, 1 - self.alpha, hist, self.alpha, 0)

    def penalize_backproject(self, target_backproj, frame_hsv):
        if self.background_hist is None:
            return target_backproj
        bg_backproj = cv2.calcBackProject([frame_hsv], [0, 1], self.background_hist, [0, 180, 0, 256], 1)
        return cv2.subtract(target_backproj, bg_backproj)


class EdgeTAM_InferenceEngine:
    """
    Predictive aerospace tracking engine featuring an internal Kalman Filter 
    and Zero-th Moment Area Adaptive Scaling.
    """
    def __init__(self):
        print("[EdgeTAM] Predictive Kalman Filter & Dynamic Scaling Engine Initialized.")
        self.target_hist = None
        self.base_template = None
        self.target_template = None
        
        # --- BASE GEOMETRY PARAMETERS ---
        self.patch_size = 60         
        self.search_margin = 160     
        self.smoothing_alpha = 0.65  
        self.last_known_pos = None
        self.base_area = 1.0         

        # --- KALMAN FILTER SETUP (Constant Velocity Kinematic Model) ---
        # State vector [x, y, dx, dy] -> Position and Velocity components
        self.kf = cv2.KalmanFilter(4, 2, 0)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], dtype=np.float32)
                                             
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], dtype=np.float32)
                                              
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.4
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

    def init_target_by_point(self, frame, pt):
        px, py = pt
        h, w = frame.shape[:2]
        
        self.patch_size = 60
        self.search_margin = 160
        
        xmin = max(0, px - self.patch_size // 2)
        ymin = max(0, py - self.patch_size // 2)
        xmax = min(w, px + self.patch_size // 2)
        ymax = min(h, py + self.patch_size // 2)
        
        crop = frame[ymin:ymax, xmin:xmax]
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            return False
            
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_crop = hsv_frame[ymin:ymax, xmin:xmax]
        self.target_hist = cv2.calcHist([hsv_crop], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(self.target_hist, self.target_hist, 0, 255, cv2.NORM_MINMAX)
        
        self.base_template = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        self.target_template = self.base_template.copy()
        
        # Calculate baseline object area profile inside initialization neighborhood
        roi_sz = 120
        r_xmin, r_ymin = max(0, px - roi_sz), max(0, py - roi_sz)
        r_xmax, r_ymax = min(w, px + roi_sz), min(h, py + roi_sz)
        local_mask = cv2.inRange(hsv_frame[r_ymin:r_ymax, r_xmin:r_xmax], np.array([0, 20, 20]), np.array([180, 255, 255]))
        contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.base_area = 1200.0  
        if contours:
            for c in contours:
                c[:, :, 0] += r_xmin
                c[:, :, 1] += r_ymin
                if cv2.pointPolygonTest(c, (float(px), float(py)), False) >= 0:
                    self.base_area = max(150.0, cv2.contourArea(c))
                    break

        # Seed the Kalman Filter state registers
        self.kf.statePost = np.array([[px], [py], [0], [0]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.last_known_pos = (px, py)
        
        print(f"[EdgeTAM] Profile locked. Baseline Target Area: {self.base_area:.1f}px")
        return True

    def track_frame(self, frame, memory_manager):
        if self.target_hist is None or self.last_known_pos is None:
            return None, 0.0, None
            
        h, w = frame.shape[:2]
        
        # --- UPGRADE 1: KALMAN INERTIAL KINEMATIC PREDICTION ---
        # Project the target's position based on its last tracked velocity vector
        prediction = self.kf.predict()
        pred_cx = max(0, min(w, int(prediction[0][0])))
        pred_cy = max(0, min(h, int(prediction[1][0])))
        
        # Center the localized search window around the prediction matrix path
        roi_xmin = max(0, pred_cx - self.search_margin)
        roi_ymin = max(0, pred_cy - self.search_margin)
        roi_xmax = min(w, pred_cx + self.search_margin)
        roi_ymax = min(h, pred_cy + self.search_margin)
        
        roi_frame = frame[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
        if roi_frame.shape[0] < self.patch_size or roi_frame.shape[1] < self.patch_size:
            roi_frame = frame
            roi_xmin, roi_ymin = 0, 0

        # Perform Hybrid Feature Extraction Matrix Matching
        hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        color_backproj = cv2.calcBackProject([hsv_roi], [0, 1], self.target_hist, [0, 180, 0, 256], 1)
        clean_color_map = memory_manager.penalize_backproject(color_backproj, hsv_roi)
        
        gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        edge_map = cv2.matchTemplate(gray_roi, self.target_template, cv2.TM_CCOEFF_NORMED)
        edge_map = cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        edge_map = cv2.resize(edge_map, (clean_color_map.shape[1], clean_color_map.shape[0]))
        
        fused_response = cv2.bitwise_and(clean_color_map, edge_map)
        _, max_val, _, max_loc = cv2.minMaxLoc(fused_response)
        
        raw_cx = max_loc[0] + self.patch_size // 2 + roi_xmin
        raw_cy = max_loc[1] + self.patch_size // 2 + roi_ymin
        
        smoothed_cx = int(self.smoothing_alpha * raw_cx + (1 - self.smoothing_alpha) * pred_cx)
        smoothed_cy = int(self.smoothing_alpha * raw_cy + (1 - self.smoothing_alpha) * pred_cy)
        
        normalized_confidence = max_val / 255.0
        mask_overlay = np.zeros_like(frame, dtype=np.uint8)
        tracked_area = self.base_area

        # Active Signal Verification Threshold Cutoff
        if normalized_confidence > 0.16:
            # --- KALMAN CORRECTION COVARIANCE STEP ---
            # Correct structural prediction equations with true verified visual feedback measurements
            measurement = np.array([[float(raw_cx)], [float(raw_cy)]], dtype=np.float32)
            self.kf.correct(measurement)
            self.last_known_pos = (smoothed_cx, smoothed_cy)
            
            # Extract current target shape profile
            try:
                local_mask = cv2.inRange(hsv_roi, np.array([0, 20, 20]), np.array([180, 255, 255]))
                contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    for c in contours:
                        if cv2.contourArea(c) > 100:
                            c[:, :, 0] += roi_xmin
                            c[:, :, 1] += roi_ymin
                            if cv2.pointPolygonTest(c, (float(smoothed_cx), float(smoothed_cy)), False) >= 0:
                                cv2.drawContours(mask_overlay, [c], -1, (0, 255, 0), -1)
                                tracked_area = cv2.contourArea(c)
                                break
            except:
                pass
            
            # --- UPGRADE 2: DYNAMIC ZERO-TH MOMENT SCALE ADAPTATION ---
            # Calculate geometric scale variance factor using area matrices
            scale_factor = np.sqrt(tracked_area / self.base_area)
            scale_factor = np.clip(scale_factor, 0.4, 2.5) # Boundaries to safeguard target structural limits
            
            # Dynamically rescale tracking patch parameters
            self.patch_size = int(60 * scale_factor)
            self.patch_size = max(24, min(150, self.patch_size))
            if self.patch_size % 2 != 0: 
                self.patch_size += 1
                
            # Resize original signature matrix to prevent calculation dimensions errors
            self.target_template = cv2.resize(self.base_template, (self.patch_size, self.patch_size))
            self.search_margin = int(160 * scale_factor)
            
            # Pass details to update lookalike background filters
            t_box = (max(0, smoothed_cx - self.patch_size//2), max(0, smoothed_cy - self.patch_size//2), self.patch_size, self.patch_size)
            memory_manager.update_background(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), t_box)
            
            return (smoothed_cx, smoothed_cy), normalized_confidence, mask_overlay
        else:
            # OCCLUSION FALLBACK: target signal lost. 
            # Trust the pure Kalman motion calculations to coast the search path along the estimated trajectory
            self.last_known_pos = (pred_cx, pred_cy)
            return (pred_cx, pred_cy), normalized_confidence, None


# --- Global Mouse Callback Link ---
clicked_point = None
new_point_registered = False

def mouse_click_handler(event, x, y, flags, param):
    global clicked_point, new_point_registered
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        new_point_registered = True


def main():
    global clicked_point, new_point_registered
    print("[INIT] Launching Predictive Scale-Invariant EdgeTAM Suite...")
    
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("[CRITICAL ERROR] Webcam stream failed to initialize.")
        return

    tracker_engine = EdgeTAM_InferenceEngine()
    memory_manager = DAM4SAM_MemoryManager()
    
    window_name = "EdgeTAM + DAM4SAM Scale-Predictive System"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click_handler)
    
    active_tracking_target = None
    last_printed_coord = None

    print("\n=== PREDICTIVE FLIGHT STREAM ONLINE ===")
    print("-> Click target to engage Kalman-gated scale invariant loops.")
    print("-> Press 'q' to terminate application.\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
            
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        
        if new_point_registered:
            success = tracker_engine.init_target_by_point(frame, clicked_point)
            if success:
                active_tracking_target = clicked_point
                last_printed_coord = clicked_point 
            new_point_registered = False
            
        if active_tracking_target is not None:
            start_time = time.time()
            
            tracked_coordinate, confidence, object_mask = tracker_engine.track_frame(frame, memory_manager)
            
            if tracked_coordinate and confidence > 0.16:
                active_tracking_target = tracked_coordinate
                tx, ty = tracked_coordinate
                
                if last_printed_coord is None or (abs(tx - last_printed_coord[0]) > 2 or abs(ty - last_printed_coord[1]) > 2):
                    print(f"[PREDICTIVE TELEMETRY] X: {tx:<3} | Y: {ty:<3} | Scale Patch: {tracker_engine.patch_size}x{tracker_engine.patch_size} | Signal: {confidence:.2f}")
                    last_printed_coord = (tx, ty)
                
                # Overlay the continuous target shape mask
                if object_mask is not None and np.any(object_mask):
                    display_frame = cv2.addWeighted(display_frame, 1.0, object_mask, 0.30, 0)
                
                # Draw the predictive search window (Orange Box)
                margin = tracker_engine.search_margin
                cv2.rectangle(display_frame, (tx - margin, ty - margin), (tx + margin, ty + margin), (0, 120, 255), 1)
                
                # Crosshair locking pointers
                cv2.drawMarker(display_frame, (tx, ty), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
                cv2.putText(display_frame, f"TRACK SCALE: {tracker_engine.patch_size}px", (tx - 45, ty - tracker_engine.patch_size//2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)
            else:
                # If target is temporarily lost, render prediction trajectory state configurations
                tx, ty = tracker_engine.last_known_pos
                margin = tracker_engine.search_margin
                cv2.rectangle(display_frame, (tx - margin, ty - margin), (tx + margin, ty + margin), (0, 0, 255), 2)
                cv2.putText(display_frame, "SIGNAL OCCLUSION / COASTING ON KALMAN ESTIMATE", (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            fps = int(1.0 / (time.time() - start_time)) if (time.time() - start_time) > 0 else 0
            cv2.putText(display_frame, f"STATUS: PREDICTIVE SCALE-ADAPTIVE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"LOOP SPEED: {fps} FPS", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(display_frame, "STATUS: CLICK OBJECT TO ANCHOR TRACKER", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(window_name, display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Operational runtime pipeline disconnected.")

if __name__ == "__main__":
    main()