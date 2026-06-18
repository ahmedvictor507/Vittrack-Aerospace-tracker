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
        self.ambiguity_threshold = 0.82

    def check_and_register_distractors(self, current_frame, found_center, max_val):
        if 0.50 < max_val < self.ambiguity_threshold:
            cx, cy = found_center
            h, w = current_frame.shape[:2]
            dx = max(0, cx - 100) if cx > w//2 else min(w-40, cx + 100)
            dy = max(0, cy - 100) if cy > h//2 else min(h-40, cy + 100)
            
            distractor = current_frame[dy:dy+40, dx:dx+40]
            if distractor.shape[0] == 40 and distractor.shape[1] == 40:
                self.distractor_patches.append(cv2.cvtColor(distractor, cv2.COLOR_BGR2GRAY))
                if len(self.distractor_patches) > self.max_distractors:
                    self.distractor_patches.pop(0)

    def apply_distractor_penalty(self, response_map, frame_gray):
        if not self.distractor_patches:
            return response_map
            
        for dist_patch in self.distractor_patches:
            try:
                dist_res = cv2.matchTemplate(frame_gray, dist_patch, cv2.TM_CCOEFF_NORMED)
                dist_res = cv2.resize(dist_res, (response_map.shape[1], response_map.shape[0]))
                response_map = np.clip(response_map - (dist_res * 0.25), 0, 1)
            except:
                pass
        return response_map


class EdgeTAM_InferenceEngine:
    """
    Real-time spatial matching engine. Emulates EdgeTAM's point-to-patch
    tracking mechanism on localized CPU environments.
    """
    def __init__(self):
        print("[EdgeTAM] High-speed visual feature tracking engine initialized.")
        self.target_patch = None
        self.patch_size = 40  

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
            print(f"[EdgeTAM] Feature lock acquired on tracking coordinate token: X={px}, Y={py}")
            return True
        return False

    def track_frame(self, frame, memory_manager):
        if self.target_patch is None:
            return None, 0.0
            
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        response_map = cv2.matchTemplate(frame_gray, self.target_patch, cv2.TM_CCOEFF_NORMED)
        response_map = memory_manager.apply_distractor_penalty(response_map, frame_gray)
        
        _, max_val, _, max_loc = cv2.minMaxLoc(response_map)
        
        cx = max_loc[0] + self.patch_size // 2
        cy = max_loc[1] + self.patch_size // 2
        
        memory_manager.check_and_register_distractors(frame, (cx, cy), max_val)
        return (cx, cy), max_val


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
    print("[INIT] Launching Real-Time Point EdgeTAM Pipeline...")
    
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("[CRITICAL ERROR] Webcam stream failed to initialize.")
        return

    tracker_engine = EdgeTAM_InferenceEngine()
    memory_manager = DAM4SAM_MemoryManager()
    
    window_name = "EdgeTAM + DAM4SAM Active Point Suite"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click_handler)
    
    active_tracking_target = None
    
    # Telemetry State: Used to detect if the target actually shifted positions
    last_printed_coord = None

    print("\n=== LIVE STREAMING ONLINE ===")
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
                last_printed_coord = clicked_point # Reset tracking telemetry base
            new_point_registered = False
            
        if active_tracking_target is not None:
            start_time = time.time()
            
            tracked_coordinate, confidence = tracker_engine.track_frame(frame, memory_manager)
            
            if tracked_coordinate and confidence > 0.35:
                active_tracking_target = tracked_coordinate
                tx, ty = tracked_coordinate
                
                # --- LIVE TERMINAL COORDINATE LOGGING ---
                # Check if the object moved more than 2 pixels away from its last known logged position
                if last_printed_coord is None or (abs(tx - last_printed_coord[0]) > 2 or abs(ty - last_printed_coord[1]) > 2):
                    print(f"[MOVEMENT DETECTED] X: {tx:<3} | Y: {ty:<3} | Confidence: {confidence:.2f}")
                    last_printed_coord = (tx, ty)
                
                # --- VISUAL RENDERING BLOCK ---
                # 1. Target Intersection Crosshair
                cv2.drawMarker(display_frame, (tx, ty), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                # 2. Tracking Area Patch Neighborhood Visual
                cv2.rectangle(display_frame, (tx - 20, ty - 20), (tx + 20, ty + 20), (255, 0, 0), 2)
                # 3. Dynamic Text Label Tag on Screen
                cv2.putText(display_frame, f"X: {tx}, Y: {ty}", (tx - 35, ty - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            else:
                cv2.putText(display_frame, "TARGET LOST / LOW CONFIDENCE", (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            fps = int(1.0 / (time.time() - start_time)) if (time.time() - start_time) > 0 else 0
            cv2.putText(display_frame, f"STATUS: TRACKING ACTIVE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"PROCESSING SPEED: {fps} FPS", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
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