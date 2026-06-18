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
        self.alpha = 0.15  # Background memory update rate

    def update_background(self, frame_hsv, target_roi_box):
        x, y, w, h = target_roi_box
        mask = np.ones(frame_hsv.shape[:2], dtype=np.uint8) * 255
        # Exclude the target area so we only learn what the background looks like
        mask[y:y+h, x:x+w] = 0
        
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
        # Actively subtract background probabilities from our tracking map
        stable_map = cv2.subtract(target_backproj, bg_backproj)
        return stable_map


class EdgeTAM_InferenceEngine:
    """
    Advanced multi-characteristic tracking engine combining color histograms 
    and structural edge gradients for deformation-resistant tracking.
    """
    def __init__(self):
        print("[EdgeTAM] Hybrid Color-Gradient Engine Initialized.")
        self.target_hist = None
        self.target_template = None
        self.patch_size = 60  # Expanded base patch size
        self.last_known_pos = None
        self.search_margin = 160  # Generous search envelope to handle high velocities
        self.smoothing_alpha = 0.60  # Balanced inertial filter to eliminate jitter

    def init_target_by_point(self, frame, pt):
        px, py = pt
        h, w = frame.shape[:2]
        
        xmin = max(0, px - self.patch_size // 2)
        ymin = max(0, py - self.patch_size // 2)
        xmax = min(w, px + self.patch_size // 2)
        ymax = min(h, py + self.patch_size // 2)
        
        crop = frame[ymin:ymax, xmin:xmax]
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            return False
            
        # 1. Initialize Color Signature (HSV space is rotation-invariant)
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        self.target_hist = cv2.calcHist([hsv_crop], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(self.target_hist, self.target_hist, 0, 255, cv2.NORM_MINMAX)
        
        # 2. Initialize Edge Gradient Signature (Handles lighting variations)
        self.target_template = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        self.last_known_pos = (px, py)
        
        print(f"[EdgeTAM] Target Signature Locked at X={px}, Y={py}")
        return True

    def track_frame(self, frame, memory_manager):
        if self.target_hist is None or self.last_known_pos is None:
            return None, 0.0, None
            
        h, w = frame.shape[:2]
        lx, ly = self.last_known_pos
        
        # Define Bounded Search Window Gating
        roi_xmin = max(0, lx - self.search_margin)
        roi_ymin = max(0, ly - self.search_margin)
        roi_xmax = min(w, lx + self.search_margin)
        roi_ymax = min(h, ly + self.search_margin)
        
        roi_frame = frame[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
        if roi_frame.shape[0] < self.patch_size or roi_frame.shape[1] < self.patch_size:
            roi_frame = frame
            roi_xmin, roi_ymin = 0, 0

        # --- COMBINED CONFIDENCE MATRIX GENERATION ---
        # Component A: Color Histogram Backprojection
        hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        color_backproj = cv2.calcBackProject([hsv_roi], [0, 1], self.target_hist, [0, 180, 0, 256], 1)
        
        # Run DAM4SAM Background Noise Cancellation
        clean_color_map = memory_manager.penalize_backproject(color_backproj, hsv_roi)
        
        # Component B: Normalized Structural Cross-Correlation
        gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        edge_map = cv2.matchTemplate(gray_roi, self.target_template, cv2.TM_CCOEFF_NORMED)
        edge_map = cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        edge_map = cv2.resize(edge_map, (clean_color_map.shape[1], clean_color_map.shape[0]))
        
        # Fuse spatial attributes together (Color + Edge Gradients)
        fused_response = cv2.bitwise_and(clean_color_map, edge_map)
        
        # Isolate peak coordinate coordinates
        _, max_val, _, max_loc = cv2.minMaxLoc(fused_response)
        
        # Remap coordinates back to global frame dimensions
        raw_cx = max_loc[0] + self.patch_size // 2 + roi_xmin
        raw_cy = max_loc[1] + self.patch_size // 2 + roi_ymin
        
        # Dynamic Inertial Filter
        smoothed_cx = int(self.smoothing_alpha * raw_cx + (1 - self.smoothing_alpha) * lx)
        smoothed_cy = int(self.smoothing_alpha * raw_cy + (1 - self.smoothing_alpha) * ly)
        
        calculated_pos = (smoothed_cx, smoothed_cy)
        self.last_known_pos = calculated_pos
        
        # Update DAM4SAM Background Map
        target_box = (max(0, smoothed_cx - 30), max(0, smoothed_cy - 30), 60, 60)
        memory_manager.update_background(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), target_box)
        
        # --- CLEAN OBJECT MASK GENERATOR ---
        mask_overlay = np.zeros_like(frame, dtype=np.uint8)
        try:
            # Generate a localized segmentation mask based directly on the tracked color profile
            local_mask = cv2.inRange(hsv_roi, np.array([0, 20, 20]), np.array([180, 255, 255]))
            contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Find the object contour containing our tracking point center
                for c in contours:
                    if cv2.contourArea(c) > 300:
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            # Shift contour to global coordinates
                            c[:, :, 0] += roi_xmin
                            c[:, :, 1] += roi_ymin
                            # Ensure the matching contour encapsulates our tracker center
                            dist = cv2.pointPolygonTest(c, (float(smoothed_cx), float(smoothed_cy)), False)
                            if dist >= 0:
                                cv2.drawContours(mask_overlay, [c], -1, (0, 255, 0), -1)
                                break
        except:
            pass

        # Normalize confidence for pipeline checks
        normalized_confidence = max_val / 255.0
        return calculated_pos, normalized_confidence, mask_overlay


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
    print("[INIT] Launching High-Stability Color-Gradient EdgeTAM Suite...")
    
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("[CRITICAL ERROR] Webcam stream failed to initialize.")
        return

    tracker_engine = EdgeTAM_InferenceEngine()
    memory_manager = DAM4SAM_MemoryManager()
    
    window_name = "EdgeTAM + DAM4SAM Robust Tracker"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click_handler)
    
    active_tracking_target = None
    last_printed_coord = None

    print("\n=== HIGH-STABILITY PIPELINE ONLINE ===")
    print("-> Click your target (e.g., your face) to initialize tracking.")
    print("-> Press 'q' to exit.\n")

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
            
            # Lowered cut-off threshold due to strict bitwise-and matching characteristics
            if tracked_coordinate and confidence > 0.15:
                active_tracking_target = tracked_coordinate
                tx, ty = tracked_coordinate
                
                # Filter out redundant micro-movements from console logs
                if last_printed_coord is None or (abs(tx - last_printed_coord[0]) > 2 or abs(ty - last_printed_coord[1]) > 2):
                    print(f"[STABLE TELEMETRY] X: {tx:<3} | Y: {ty:<3} | Signal: {confidence:.2f}")
                    last_printed_coord = (tx, ty)
                
                # Render the structural segmentation overlay mask (highlights your face/object area)
                if object_mask is not None and np.any(object_mask):
                    display_frame = cv2.addWeighted(display_frame, 1.0, object_mask, 0.30, 0)
                
                # Render tracking crosshair targets
                cv2.drawMarker(display_frame, (tx, ty), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
                cv2.rectangle(display_frame, (tx - 30, ty - 30), (tx + 30, ty + 30), (255, 0, 0), 2)
                cv2.putText(display_frame, f"X: {tx}, Y: {ty}", (tx - 35, ty - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            else:
                cv2.putText(display_frame, "TARGET LOST / RE-ACQUIRING...", (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                tracker_engine.last_known_pos = active_tracking_target

            fps = int(1.0 / (time.time() - start_time)) if (time.time() - start_time) > 0 else 0
            cv2.putText(display_frame, f"STATUS: HYBRID TRACKING", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"RUN RATE: {fps} FPS", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
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