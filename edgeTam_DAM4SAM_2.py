import cv2
import numpy as np
import time

class DAM4SAM_MemoryManager:
    """
    Manages lookalike background distractor profiles.
    If an area outside our target matches too closely, we penalize it.
    """
    def __init__(self):
        self.distractor_patches = []
        self.max_distractors = 3
        self.ambiguity_threshold = 0.85

    def check_and_register_distractors(self, current_frame, found_center, max_val):
        if 0.50 < max_val < self.ambiguity_threshold:
            cx, cy = found_center
            h, w = current_frame.shape[:2]
            dx = max(0, cx - 80) if cx > w//2 else min(w-40, cx + 80)
            dy = max(0, cy - 80) if cy > h//2 else min(h-40, cy + 80)
            
            distractor = current_frame[dy:dy+40, dx:dx+40]
            if distractor.shape[0] == 40 and distractor.shape[1] == 40:
                self.distractor_patches.append(cv2.cvtColor(distractor, cv2.COLOR_BGR2GRAY))
                if len(self.distractor_patches) > self.max_distractors:
                    self.distractor_patches.pop(0)

    def apply_distractor_penalty(self, response_map, frame_gray_roi):
        if not self.distractor_patches:
            return response_map
            
        for dist_patch in self.distractor_patches:
            try:
                # Evaluate matches specifically within our tracking neighborhood
                dist_res = cv2.matchTemplate(frame_gray_roi, dist_patch, cv2.TM_CCOEFF_NORMED)
                dist_res = cv2.resize(dist_res, (response_map.shape[1], response_map.shape[0]))
                response_map = np.clip(response_map - (dist_res * 0.30), 0, 1)
            except:
                pass
        return response_map


class EdgeTAM_InferenceEngine:
    """
    Flight-stabilized visual feature engine using Localized Search Gating
    to prevent target jumping and coordinate snapping.
    """
    def __init__(self):
        print("[EdgeTAM] Localized Flight-Stabilization Tracker Active.")
        self.target_patch = None
        self.patch_size = 40  
        self.last_known_pos = None
        
        # DRONE TUNING PARAMETERS:
        self.search_margin = 70  # Radius around the target to search (prevents global jumping)
        self.smoothing_alpha = 0.25  # Lower = smoother/buffered, Higher = raw/twitchy response

    def init_target_by_point(self, frame, pt):
        px, py = pt
        h, w = frame.shape[:2]
        
        xmin = max(0, px - self.patch_size // 2)
        ymin = max(0, py - self.patch_size // 2)
        xmax = min(w, px + self.patch_size // 2)
        ymax = min(h, py + self.patch_size // 2)
        
        crop = frame[ymin:ymax, xmin:xmax]
        if crop.shape[0] == self.patch_size and crop.shape[1] == self.patch_size:
            self.target_patch = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            self.last_known_pos = (px, py)
            print(f"[EdgeTAM] Target locked at: X={px}, Y={py}. Bounded search window active.")
            return True
        return False

    def track_frame(self, frame, memory_manager):
        if self.target_patch is None or self.last_known_pos is None:
            return None, 0.0
            
        h, w = frame.shape[:2]
        lx, ly = self.last_known_pos
        
        # 1. LOCALIZED SEARCH WINDOW CROP (GATING)
        # Calculate search bounding box limits around last known coordinates
        roi_xmin = max(0, lx - self.search_margin)
        roi_ymin = max(0, ly - self.search_margin)
        roi_xmax = min(w, lx + self.search_margin)
        roi_ymax = min(h, ly + self.search_margin)
        
        roi_frame = frame[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
        if roi_frame.shape[0] < self.patch_size or roi_frame.shape[1] < self.patch_size:
            # Fallback to global frame search if the window breaks down at boundary edges
            roi_frame = frame
            roi_xmin, roi_ymin = 0, 0

        frame_gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        
        # Match features only inside our restricted search bubble
        response_map = cv2.matchTemplate(frame_gray_roi, self.target_patch, cv2.TM_CCOEFF_NORMED)
        response_map = memory_manager.apply_distractor_penalty(response_map, frame_gray_roi)
        
        _, max_val, _, max_loc = cv2.minMaxLoc(response_map)
        
        # 2. COORDINATE REMAPPING & EXPONENTIAL FILTERING
        # Translate local crop coordinates back up to global screen coordinate mappings
        raw_cx = max_loc[0] + self.patch_size // 2 + roi_xmin
        raw_cy = max_loc[1] + self.patch_size // 2 + roi_ymin
        
        # Apply the Exponential Moving Average filter to smooth drone control variables
        smoothed_cx = int(self.smoothing_alpha * raw_cx + (1 - self.smoothing_alpha) * lx)
        smoothed_cy = int(self.smoothing_alpha * raw_cy + (1 - self.smoothing_alpha) * ly)
        
        calculated_pos = (smoothed_cx, smoothed_cy)
        self.last_known_pos = calculated_pos
        
        memory_manager.check_and_register_distractors(frame, calculated_pos, max_val)
        return calculated_pos, max_val


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
    print("[INIT] Launching Flight-Stabilized EdgeTAM Pipeline...")
    
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("[CRITICAL ERROR] Webcam stream failed to initialize.")
        return

    tracker_engine = EdgeTAM_InferenceEngine()
    memory_manager = DAM4SAM_MemoryManager()
    
    window_name = "EdgeTAM + DAM4SAM Flight Tracker"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click_handler)
    
    active_tracking_target = None
    last_printed_coord = None

    print("\n=== STABILIZED STREAM ONLINE ===")
    print("-> Click ANY pixel on the video screen to lock tracking.")
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
            
            tracked_coordinate, confidence = tracker_engine.track_frame(frame, memory_manager)
            
            # Flight-ready tracking cutoff threshold
            if tracked_coordinate and confidence > 0.40:
                active_tracking_target = tracked_coordinate
                tx, ty = tracked_coordinate
                
                # Active movement detection telemetry logger
                if last_printed_coord is None or (abs(tx - last_printed_coord[0]) > 1 or abs(ty - last_printed_coord[1]) > 1):
                    print(f"[FLIGHT COORDS] X: {tx:<3} | Y: {ty:<3} | Signal: {confidence:.2f}")
                    last_printed_coord = (tx, ty)
                
                # Draw the local search boundary to visually inspect stability
                margin = tracker_engine.search_margin
                cv2.rectangle(display_frame, (tx - margin, ty - margin), (tx + margin, ty + margin), (0, 120, 255), 1)
                
                # Draw visual center point and marker
                cv2.drawMarker(display_frame, (tx, ty), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
                cv2.rectangle(display_frame, (tx - 20, ty - 20), (tx + 20, ty + 20), (255, 0, 0), 2)
                cv2.putText(display_frame, f"X: {tx}, Y: {ty}", (tx - 35, ty - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            else:
                cv2.putText(display_frame, "TARGET LOST / SEARCHING", (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # Fallback target position to trigger wider re-acquisition
                tracker_engine.last_known_pos = active_tracking_target

            fps = int(1.0 / (time.time() - start_time)) if (time.time() - start_time) > 0 else 0
            cv2.putText(display_frame, f"STATUS: TRACKING ACTIVE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"LOOP SPEED: {fps} FPS", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(display_frame, "STATUS: CLICK OBJECT TO ANCHOR TRACKER", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(window_name, display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Operational runtime pipeline successfully disconnected.")

if __name__ == "__main__":
    main()