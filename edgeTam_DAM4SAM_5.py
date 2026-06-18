import cv2
import numpy as np
import time

class DAM4SAM_MemoryManager:
    """
    Suppresses background environmental noise and lookalike distractors.
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
    Advanced Multi-Domain Engine: Asymmetric Kalman Filtering, Aspect Ratio Morphing,
    and Sensor-Specific (Day/Night) Inference Modes.
    """
    def __init__(self):
        print("[EdgeTAM] Multi-Domain Engine Initialized (Asymmetric Kalman + Aspect Morphing).")
        self.target_hist = None
        self.base_template = None
        self.target_template = None
        
        # --- SENSOR MODE ---
        self.night_mode = False
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) # Contrast enhancer for IR/Thermal
        
        # --- DYNAMIC GEOMETRY (ASPECT RATIO AWARE) ---
        self.patch_w = 60         
        self.patch_h = 60
        self.search_margin = 160     
        self.smoothing_alpha = 0.65  
        self.last_known_pos = None
        self.base_area = 1200.0         

        # --- ASYMMETRIC KALMAN FILTER (Drone Flight Dynamics) ---
        self.kf = cv2.KalmanFilter(4, 2, 0)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], dtype=np.float32)
                                              
        # Asymmetric Process Noise: X (Yaw) is agile [0.05], Y (Pitch/Altitude) is rigid [0.01]
        self.kf.processNoiseCov = np.array([[0.05, 0, 0, 0],
                                            [0, 0.01, 0, 0],
                                            [0, 0, 0.05, 0],
                                            [0, 0, 0, 0.01]], dtype=np.float32)
                                            
        # Asymmetric Measurement Noise: Trust X visual data more, heavily filter Y jitter
        self.kf.measurementNoiseCov = np.array([[0.2, 0],
                                                [0, 0.6]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

    def toggle_night_mode(self):
        self.night_mode = not self.night_mode
        state = "NIGHT/THERMAL (Grayscale CLAHE)" if self.night_mode else "DAY (Hybrid Color-Gradient)"
        print(f"\n[SENSOR OVERRIDE] Switching Tracking Profile -> {state}")

    def init_target_by_point(self, frame, pt):
        px, py = pt
        h, w = frame.shape[:2]
        
        self.patch_w = 60
        self.patch_h = 60
        self.search_margin = 160
        
        xmin = max(0, px - self.patch_w // 2)
        ymin = max(0, py - self.patch_h // 2)
        xmax = min(w, px + self.patch_w // 2)
        ymax = min(h, py + self.patch_h // 2)
        
        crop = frame[ymin:ymax, xmin:xmax]
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            return False
            
        # Extract visual signatures
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_crop = hsv_frame[ymin:ymax, xmin:xmax]
        self.target_hist = cv2.calcHist([hsv_crop], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(self.target_hist, self.target_hist, 0, 255, cv2.NORM_MINMAX)
        
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Apply Contrast Limited Adaptive Histogram Equalization if in night mode
        self.base_template = self.clahe.apply(gray_crop) if self.night_mode else gray_crop
        self.target_template = self.base_template.copy()
        
        # Calculate baseline object area profile
        roi_sz = 120
        r_xmin, r_ymin = max(0, px - roi_sz), max(0, py - roi_sz)
        r_xmax, r_ymax = min(w, px + roi_sz), min(h, py + roi_sz)
        
        # Mode-specific contour generation
        if self.night_mode:
            local_gray = cv2.cvtColor(frame[r_ymin:r_ymax, r_xmin:r_xmax], cv2.COLOR_BGR2GRAY)
            local_gray = self.clahe.apply(local_gray)
            local_mask = cv2.adaptiveThreshold(local_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        else:
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

        self.kf.statePost = np.array([[px], [py], [0], [0]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.last_known_pos = (px, py)
        
        print(f"[EdgeTAM] Target locked. Area: {self.base_area:.1f}px")
        return True

    def track_frame(self, frame, memory_manager):
        if self.target_hist is None or self.last_known_pos is None:
            return None, 0.0, None
            
        h, w = frame.shape[:2]
        
        prediction = self.kf.predict()
        pred_cx = max(0, min(w, int(prediction[0][0])))
        pred_cy = max(0, min(h, int(prediction[1][0])))
        
        roi_xmin = max(0, pred_cx - self.search_margin)
        roi_ymin = max(0, pred_cy - self.search_margin)
        roi_xmax = min(w, pred_cx + self.search_margin)
        roi_ymax = min(h, pred_cy + self.search_margin)
        
        roi_frame = frame[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
        if roi_frame.shape[0] < self.patch_h or roi_frame.shape[1] < self.patch_w:
            roi_frame = frame
            roi_xmin, roi_ymin = 0, 0

        # --- SENSOR-SPECIFIC FEATURE EXTRACTION ---
        gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        
        if self.night_mode:
            # Thermal / IR / Night Mode: Bypass color, use CLAHE enhanced structural edge mapping
            enhanced_gray = self.clahe.apply(gray_roi)
            fused_response = cv2.matchTemplate(enhanced_gray, self.target_template, cv2.TM_CCOEFF_NORMED)
            fused_response = cv2.normalize(fused_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            # Day Mode: Hybrid Color + Edge Map
            hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
            color_backproj = cv2.calcBackProject([hsv_roi], [0, 1], self.target_hist, [0, 180, 0, 256], 1)
            clean_color_map = memory_manager.penalize_backproject(color_backproj, hsv_roi)
            
            edge_map = cv2.matchTemplate(gray_roi, self.target_template, cv2.TM_CCOEFF_NORMED)
            edge_map = cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            edge_map = cv2.resize(edge_map, (clean_color_map.shape[1], clean_color_map.shape[0]))
            fused_response = cv2.bitwise_and(clean_color_map, edge_map)
            
        _, max_val, _, max_loc = cv2.minMaxLoc(fused_response)
        
        raw_cx = max_loc[0] + self.patch_w // 2 + roi_xmin
        raw_cy = max_loc[1] + self.patch_h // 2 + roi_ymin
        
        smoothed_cx = int(self.smoothing_alpha * raw_cx + (1 - self.smoothing_alpha) * pred_cx)
        smoothed_cy = int(self.smoothing_alpha * raw_cy + (1 - self.smoothing_alpha) * pred_cy)
        
        normalized_confidence = max_val / 255.0
        mask_overlay = np.zeros_like(frame, dtype=np.uint8)
        
        confidence_threshold = 0.40 if self.night_mode else 0.16

        if normalized_confidence > confidence_threshold:
            measurement = np.array([[float(raw_cx)], [float(raw_cy)]], dtype=np.float32)
            self.kf.correct(measurement)
            self.last_known_pos = (smoothed_cx, smoothed_cy)
            
            target_contour = None
            try:
                if self.night_mode:
                    local_mask = cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                else:
                    local_mask = cv2.inRange(hsv_roi, np.array([0, 20, 20]), np.array([180, 255, 255]))
                    
                contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    for c in contours:
                        if cv2.contourArea(c) > 15:
                            c[:, :, 0] += roi_xmin
                            c[:, :, 1] += roi_ymin
                            if cv2.pointPolygonTest(c, (float(smoothed_cx), float(smoothed_cy)), False) >= 0:
                                mask_color = (0, 100, 255) if self.night_mode else (0, 255, 0)
                                cv2.drawContours(mask_overlay, [c], -1, mask_color, -1)
                                target_contour = c
                                break
            except:
                pass
            
            # --- ASPECT RATIO MORPHING & ADAPTIVE SCALING ---
            if target_contour is not None:
                cx, cy, cw, ch = cv2.boundingRect(target_contour)
                # Dynamic aspect ratio evaluation
                morph_w = max(12, min(180, int(cw * 0.85)))
                morph_h = max(12, min(180, int(ch * 0.85)))
                
                # Smooth morphing over time to prevent jittery bounding boxes
                self.patch_w = int(0.9 * self.patch_w + 0.1 * morph_w)
                self.patch_h = int(0.9 * self.patch_h + 0.1 * morph_h)
            
            # Reconstruct template geometry
            self.target_template = cv2.resize(self.base_template, (self.patch_w, self.patch_h))
            
            # Adjust search envelope based on target size
            self.search_margin = max(100, int(max(self.patch_w, self.patch_h) * 2.5))
            
            if not self.night_mode:
                t_box = (max(0, smoothed_cx - self.patch_w//2), max(0, smoothed_cy - self.patch_h//2), self.patch_w, self.patch_h)
                memory_manager.update_background(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), t_box)
            
            return (smoothed_cx, smoothed_cy), normalized_confidence, mask_overlay
        else:
            self.last_known_pos = (pred_cx, pred_cy)
            return (pred_cx, pred_cy), normalized_confidence, None


# --- Global UI Controls ---
clicked_point = None
new_point_registered = False

def mouse_click_handler(event, x, y, flags, param):
    global clicked_point, new_point_registered
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        new_point_registered = True


def main():
    global clicked_point, new_point_registered
    print("[INIT] Launching Multi-Domain Aspect-Morphing EdgeTAM Suite...")
    
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("[CRITICAL ERROR] Webcam stream failed.")
        return

    tracker_engine = EdgeTAM_InferenceEngine()
    memory_manager = DAM4SAM_MemoryManager()
    
    window_name = "EdgeTAM Multi-Domain Track"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click_handler)
    
    active_tracking_target = None

    print("\n=== SYSTEMS ONLINE ===")
    print("-> Click target to engage.")
    print("-> Press 'n' to toggle DAY / NIGHT(THERMAL) mode.")
    print("-> Press 'q' to terminate.\n")

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
            new_point_registered = False
            
        if active_tracking_target is not None:
            start_time = time.time()
            tracked_coordinate, confidence, object_mask = tracker_engine.track_frame(frame, memory_manager)
            
            if object_mask is not None and np.any(object_mask):
                display_frame = cv2.addWeighted(display_frame, 1.0, object_mask, 0.40, 0)
            
            cut_off = 0.40 if tracker_engine.night_mode else 0.16
            if tracked_coordinate and confidence > cut_off:
                active_tracking_target = tracked_coordinate
                tx, ty = tracked_coordinate
                
                # Draw morphing aspect ratio tracking box
                pw, ph = tracker_engine.patch_w // 2, tracker_engine.patch_h // 2
                cv2.rectangle(display_frame, (tx - pw, ty - ph), (tx + pw, ty + ph), (255, 255, 0), 2)
                cv2.drawMarker(display_frame, (tx, ty), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
                
                cv2.putText(display_frame, f"AR: {tracker_engine.patch_w}x{tracker_engine.patch_h}", 
                            (tx - 40, ty - ph - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1)
            else:
                tx, ty = tracker_engine.last_known_pos
                pw, ph = tracker_engine.patch_w // 2, tracker_engine.patch_h // 2
                cv2.rectangle(display_frame, (tx - pw, ty - ph), (tx + pw, ty + ph), (0, 0, 255), 2)
                cv2.putText(display_frame, "COASTING (KALMAN ONLY)", (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            fps = int(1.0 / (time.time() - start_time)) if (time.time() - start_time) > 0 else 0
            mode_str = "NIGHT/THERMAL MODE" if tracker_engine.night_mode else "DAY/COLOR MODE"
            color_str = (0, 120, 255) if tracker_engine.night_mode else (0, 255, 0)
            cv2.putText(display_frame, f"MODE: {mode_str}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_str, 2)
            cv2.putText(display_frame, f"SPEED: {fps} FPS", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(display_frame, "CLICK OBJECT TO ANCHOR", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            tracker_engine.toggle_night_mode()

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Pipeline disconnected.")

if __name__ == "__main__":
    main()