import cv2
import numpy as np
import time
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Import the AerospaceTracker from the existing script
from edgeTam_DAM4SAM_6 import AerospaceTracker

class UnifiedTrackerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Tracking Interface")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg='#1e1e1e')
        self.root.bind('<Escape>', lambda e: self.quit_app())

        # Tracker State
        self.tracker_type = tk.StringVar(value="VitTrack")
        self.tracking_active = False
        self.attack_active = False
        
        # Mouse Drag State
        self.drawing = False
        self.ix = 0
        self.iy = 0
        self.curr_x = 0
        self.curr_y = 0
        self.selection_rect_id = None
        self.image_id = None

        # Initialize Trackers
        self.vit_params = cv2.TrackerVit_Params()
        model_path = "object_tracking_vittrack_2023sep.onnx"
        if not os.path.exists(model_path):
            print(f"[ERROR] Missing {model_path}")
            self.vit_tracker = None
        else:
            self.vit_params.net = model_path
            self.vit_tracker = cv2.TrackerVit_create(self.vit_params)
            
        self.aerospace_tracker = AerospaceTracker()

        # Initialize Camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Read a single frame to get dimensions
        ret, frame = self.cap.read()
        if ret:
            self.frame_h, self.frame_w = frame.shape[:2]
        else:
            self.frame_w, self.frame_h = 640, 480 # Fallback

        self.setup_ui()
        self.update_frame()

    def setup_ui(self):
        # Top Label
        title = tk.Label(self.root, text="DRONE TRACKING CENTER", font=("Helvetica", 24, "bold"), bg='#1e1e1e', fg='#00FF00')
        title.pack(pady=10)

        # Video Canvas (centered)
        self.canvas = tk.Canvas(self.root, width=self.frame_w, height=self.frame_h, bg='black', highlightthickness=2, highlightbackground='#333333')
        self.canvas.pack()

        # Canvas Mouse Bindings
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Control Panel Frame
        control_frame = tk.Frame(self.root, bg='#1e1e1e')
        control_frame.pack(pady=20)

        # Styling
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TCombobox', font=('Helvetica', 14))

        # Tracker Selection
        tk.Label(control_frame, text="Tracker Engine:", font=("Helvetica", 14), bg='#1e1e1e', fg='white').grid(row=0, column=0, padx=10)
        self.tracker_combo = ttk.Combobox(control_frame, textvariable=self.tracker_type, values=["VitTrack", "AerospaceTracker"], state="readonly", width=20, font=("Helvetica", 14))
        self.tracker_combo.grid(row=0, column=1, padx=10)
        self.tracker_combo.bind("<<ComboboxSelected>>", self.on_tracker_change)

        # Attack Button
        self.attack_btn = tk.Button(control_frame, text="ATTACK (OFF)", font=("Helvetica", 16, "bold"), bg='#444444', fg='white', 
                                    activebackground='#ff4444', activeforeground='white', command=self.toggle_attack, width=15)
        self.attack_btn.grid(row=0, column=2, padx=30)

        # Quit Button
        self.quit_btn = tk.Button(control_frame, text="QUIT", font=("Helvetica", 16, "bold"), bg='#882222', fg='white', command=self.quit_app, width=10)
        self.quit_btn.grid(row=0, column=3, padx=10)
        
        # Info Label
        tk.Label(self.root, text="Click and drag on the video to select a target.", font=("Helvetica", 12), bg='#1e1e1e', fg='#888888').pack()

    def on_tracker_change(self, event):
        self.tracking_active = False

    def toggle_attack(self):
        self.attack_active = not self.attack_active
        if self.attack_active:
            self.attack_btn.configure(text="ATTACK (ON)", bg='red', fg='white')
        else:
            self.attack_btn.configure(text="ATTACK (OFF)", bg='#444444', fg='white')

    def on_mouse_down(self, event):
        self.tracking_active = False
        self.drawing = True
        self.ix, self.iy = event.x, event.y
        self.curr_x, self.curr_y = event.x, event.y
        if self.selection_rect_id:
            self.canvas.delete(self.selection_rect_id)
        self.selection_rect_id = self.canvas.create_rectangle(self.ix, self.iy, self.curr_x, self.curr_y, outline="cyan", width=2)

    def on_mouse_drag(self, event):
        if self.drawing:
            self.curr_x, self.curr_y = event.x, event.y
            self.canvas.coords(self.selection_rect_id, self.ix, self.iy, self.curr_x, self.curr_y)

    def on_mouse_up(self, event):
        self.drawing = False
        self.curr_x, self.curr_y = event.x, event.y
        
        x_min = min(self.ix, self.curr_x)
        y_min = min(self.iy, self.curr_y)
        w = abs(self.curr_x - self.ix)
        h = abs(self.curr_y - self.iy)
        
        self.canvas.delete(self.selection_rect_id)
        self.selection_rect_id = None

        if w > 10 and h > 10:
            ret, frame = self.cap.read()
            if not ret: return
            frame = cv2.flip(frame, 1)

            ttype = self.tracker_type.get()
            if ttype == "VitTrack" and self.vit_tracker:
                # Recreate to reset state safely
                self.vit_tracker = cv2.TrackerVit_create(self.vit_params)
                self.vit_tracker.init(frame, (x_min, y_min, w, h))
                self.tracking_active = True
            elif ttype == "AerospaceTracker":
                cx, cy = x_min + w//2, y_min + h//2
                success = self.aerospace_tracker.init_target(frame, (cx, cy))
                if success:
                    self.tracking_active = True

    def process_attack_deadband(self, cx, cy, frame):
        center_x, center_y = self.frame_w // 2, self.frame_h // 2
        err_x = cx - center_x
        err_y = cy - center_y
        
        if abs(err_x) < 15: err_x = 0
        if abs(err_y) < 15: err_y = 0
        
        if err_x != 0 or err_y != 0:
            print(f"[MAVLink SIM] ATTACKING! dx:{err_x}, dy:{err_y}")
            cv2.putText(frame, f"ATTACKING! dx:{err_x} dy:{err_y}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.line(frame, (center_x, center_y), (cx, cy), (0, 0, 255), 2)
        else:
            print("[MAVLink SIM] ATTACK LOCKED - HOLDING POSITION")
            cv2.putText(frame, "ATTACK LOCKED - DEAD CENTER", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Crosshair
            cv2.drawMarker(display_frame, (self.frame_w//2, self.frame_h//2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)

            if self.tracking_active:
                ttype = self.tracker_type.get()
                
                if ttype == "VitTrack" and self.vit_tracker:
                    success, box = self.vit_tracker.update(frame)
                    if success:
                        (x, y, w, h) = [int(v) for v in box]
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(display_frame, "VitTrack", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        if self.attack_active:
                            self.process_attack_deadband(x + w//2, y + h//2, display_frame)
                    else:
                        cv2.putText(display_frame, "TARGET LOST", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        self.tracking_active = False

                elif ttype == "AerospaceTracker":
                    pos, conf = self.aerospace_tracker.track(frame)
                    if pos:
                        cx, cy = pos
                        pw, ph = self.aerospace_tracker.patch_w // 2, self.aerospace_tracker.patch_h // 2
                        color = (0, 255, 0) if conf > 0.4 else (0, 0, 255)
                        
                        if self.aerospace_tracker.frames_lost > self.aerospace_tracker.max_lost_frames:
                            cv2.putText(display_frame, "RE-DETECTING (GLOBAL)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                            color = (0, 165, 255)
                            
                        cv2.rectangle(display_frame, (cx - pw, cy - ph), (cx + pw, cy + ph), color, 2)
                        cv2.putText(display_frame, f"AeroTracker Conf: {conf:.2f}", (cx - pw, cy - ph - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        if conf > 0.4 and self.aerospace_tracker.frames_lost <= self.aerospace_tracker.max_lost_frames:
                            if self.attack_active:
                                self.process_attack_deadband(cx, cy, display_frame)

            # Convert to Tkinter format
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            self.tk_image = ImageTk.PhotoImage(image=img)

            if self.image_id is None:
                self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
                # Ensure the selection rectangle stays on top
                if self.selection_rect_id:
                    self.canvas.tag_raise(self.selection_rect_id)
            else:
                self.canvas.itemconfig(self.image_id, image=self.tk_image)

        self.root.after(15, self.update_frame)

    def quit_app(self):
        self.cap.release()
        self.root.destroy()
        print("Application closed.")

if __name__ == "__main__":
    root = tk.Tk()
    app = UnifiedTrackerGUI(root)
    root.mainloop()
