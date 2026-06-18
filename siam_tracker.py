import cv2
import time
import os

# Globals for mouse callback
drawing = False
ix, iy = -1, -1
curr_x, curr_y = -1, -1
new_bb = None

def draw_box(event, x, y, flags, param):
    global ix, iy, curr_x, curr_y, drawing, new_bb

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        curr_x, curr_y = x, y
        new_bb = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            curr_x, curr_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        curr_x, curr_y = x, y
        x_min = min(ix, curr_x)
        y_min = min(iy, curr_y)
        w = abs(curr_x - ix)
        h = abs(curr_y - iy)
        if w > 10 and h > 10:  # Avoid accidental small clicks
            new_bb = (x_min, y_min, w, h)


def main():
    global new_bb, drawing, ix, iy, curr_x, curr_y

    print("[INIT] Booting VitTrack AI Tracking Engine...")

    # 1. Verify AI weights exist in the current directory
    required_models = [
        "object_tracking_vittrack_2023sep.onnx"
    ]
    
    for model in required_models:
        if not os.path.exists(model):
            print(f"[CRITICAL ERROR] Missing {model}!")
            print("Please download the ONNX file from OpenCV Zoo and place it into this directory.")
            print("Link: https://github.com/opencv/opencv_zoo/blob/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx")
            return

    # 2. Load the VitTrack AI Parameters
    try:
        params = cv2.TrackerVit_Params()
        params.net = "object_tracking_vittrack_2023sep.onnx"
        # Use default backend since CUDA is throwing an assertion error
        
        tracker = cv2.TrackerVit_create(params)
    except AttributeError:
        print("[ERROR] Your OpenCV version does not support VitTrack.")
        print("Please ensure you have OpenCV 4.8.0 or higher.")
        return

    # 3. Initialize Camera (Jetson Optimized)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Zero-latency buffer

    if not cap.isOpened():
        print("[CRITICAL] Camera failed to open.")
        return

    print("\n=== AI TRACKER ONLINE ===")
    print("-> Click and drag on the video to select a target directly.")
    print("-> Press 'q' to quit.\n")

    cv2.namedWindow("AI Edge Tracker")
    cv2.setMouseCallback("AI Edge Tracker", draw_box)

    init_bb = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        if new_bb is not None:
            init_bb = new_bb
            new_bb = None
            # Need to create a new tracker instance or just init again
            tracker.init(frame, init_bb)

        if drawing:
            # Draw the box while dragging
            x_min = min(ix, curr_x)
            y_min = min(iy, curr_y)
            w = abs(curr_x - ix)
            h = abs(curr_y - iy)
            cv2.rectangle(display_frame, (x_min, y_min), (x_min + w, y_min + h), (255, 0, 0), 2)
            cv2.putText(display_frame, "Selecting Target...", (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        elif init_bb is not None:
            start_time = time.time()
            
            # Feed the frame into the VitTrack Network
            success, box = tracker.update(frame)
            
            fps = int(1.0 / (time.time() - start_time)) if (time.time() - start_time) > 0 else 0

            if success:
                # Target tracked successfully
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"VitTrack | {fps} FPS", (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Check Attack Mode
                attack_mode = False
                try:
                    if os.path.exists("/tmp/drone_attack_state"):
                        with open("/tmp/drone_attack_state", "r") as f:
                            if f.read().strip() == "1":
                                attack_mode = True
                except:
                    pass

                if attack_mode:
                    target_cx = x + w // 2
                    target_cy = y + h // 2
                    frame_h, frame_w = frame.shape[:2]
                    center_x, center_y = frame_w // 2, frame_h // 2
                    
                    err_x = target_cx - center_x
                    err_y = target_cy - center_y
                    
                    if abs(err_x) < 15: err_x = 0
                    if abs(err_y) < 15: err_y = 0
                    
                    if err_x != 0 or err_y != 0:
                        print(f"[MAVLink SIM] ATTACKING! dx:{err_x}, dy:{err_y}")
                        cv2.putText(display_frame, f"ATTACKING! dx:{err_x} dy:{err_y}", (20, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        print("[MAVLink SIM] ATTACK LOCKED - HOLDING POSITION")
                        cv2.putText(display_frame, "ATTACK LOCKED - DEAD CENTER", (20, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                # Target completely lost (occluded or left frame)
                cv2.putText(display_frame, "TARGET LOST - Select new target", (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("AI Edge Tracker", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()