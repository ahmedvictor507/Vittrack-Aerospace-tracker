## Flight controller notes

If you feed raw X and Y coordinates directly into a drone's flight controller (like a Pixhawk or Betaflight board over MAVLink/ROS), the drone will constantly twitch. No matter how smooth our tracker is, micro-jitters of 1 or 2 pixels will cause the drone's motors to over-correct, burning through battery life and causing mechanical strain.

The Fix: Introduce a Deadband Zone and calculate the error vector relative to the center of the camera frame (320,240).
Errorx​=Xtarget​−320
Errory​=Ytarget​−240

If the target is within ±15 pixels of the center, the error is treated as exactly 0. The drone will only spin its motors or tilt its gimbals when the target actively attempts to leave the center crosshairs.


## edgeTam_DAM4SAM_4 notes:
It uses:
1. True Motion Prediction (The Kalman Filter)

Right now, our search window is centered exactly on the object's last known position. If your drone tilts sharply or the target moves quickly, the object can jump completely outside the search window before the tracker can update.

By integrating a Kalman Filter, you shift from reactive tracking to predictive tracking. The filter tracks position, velocity, and acceleration. Even if the target moves violently or is completely blocked by an obstacle for a few frames (temporary occlusion), the search window will automatically "guess" where the object should be and follow it.
2. Dynamic Scale Adaptation (Handling Distance)

Currently, your tracking patch size is hardcoded (e.g., 60×60 pixels).

    If the drone flies closer to you, your face gets bigger, and the tracker only looks at a tiny patch of your skin (losing context).

    If the drone drifts further away, your face gets smaller, and background noise starts bleeding into the tracking patch, causing the tracker to slip off.


## Night vision:

Night Vision (NIR) Performance

    How well it works: Highly Effective. * The Catch: Ambient night vision still captures structural textures and reflections. The Edge Gradient Matcher in the code will carry the weight beautifully because structural lines remain sharp. However, if the infrared feed switches to monochrome (black and white), the HSV Color Map loses its utility because color saturation drops to zero.

Thermal Vision (LWIR) Performance

    How well it works: Incredibly Stable (with one caveat).

    The Catch: Thermal cameras do not see light or texture; they see heat signatures. If your thermal camera outputs a raw grayscale heat map, the color matcher fails. But if your camera applies a pseudo-color palette (like "Jet" or "Ironbow" which turns heat into bright oranges and yellows), the HSV color tracker becomes an absolute weapon. It will lock onto the heat signature of a face or vehicle with almost zero background interference.

## Parameters:
Here is the complete, comprehensive breakdown of every single configuration and structural tuning parameter included in your tracking code, formatted cleanly for your documentation.

### Core Architecture Tuning Parameters

| Parameter | Type / Location | Default Value | What It Controls / How to Tune |
| --- | --- | --- | --- |
| `self.patch_size` | Feature Engine Base | `60` | The baseline pixel dimension ($60 \times 60$) of your target template. Increase for giant targets close to the camera; decrease for tiny or highly distant objects. |
| `self.search_margin` | Gating Window Base | `160` | The radius of the localized search tracking box. If your drone maneuvers violently or the target moves like a rocket, increase this to prevent the target from escaping the search field. |
| `self.smoothing_alpha` | Inertial Filter | `0.65` | The exponential smoothing weight. `1.0` means raw, instant reaction (highly reactive but noisy). `0.1` means heavily smoothed path (lagging trail, but filters out jitter). |
| `self.base_area` | Scaling Anchor | `1200.0` | The fallback pixel area used to calibrate the baseline size of the target. It acts as the denominator when calculating the scale variance factor. |
| `scale_factor` Limits | Scaler Constraint | `0.4` to `2.5` | The minimum and maximum allowable multipliers for template resizing. Prevents the bounding box from shrinking to a single pixel or growing to fill the entire screen. |
| `self.patch_size` Safety Clamp | Scaler Guard | `24` to `150` | Explicit pixel boundaries for the template patch box size. Protects against calculation dimension errors in the `cv2.matchTemplate` matrix operations. |
| Confidence Cutoff | Signal Validator | `0.16` | The hard cutoff threshold for target validation. Values below this trigger **Occlusion Mode**, letting the Kalman filter coast on pure physics instead of trusting noisy visual data. |
| Object Size Scope | Mask Component | `120` | The pixel radius mapped out during initialization to search for the boundaries of the target's continuous physical shape. |
| Contour Minimum Area | Segmentation Filter | `100` | The pixel area threshold for the contour generator. Ignores small blobs, specs of dust, or background noise smaller than 100 pixels when rendering the green mask. |

---

### Memory & Distractor Suppression Parameters

| Parameter | Type / Location | Default Value | What It Controls / How to Tune |
| --- | --- | --- | --- |
| `self.alpha` | Memory Manager | `0.15` | The learning rate for the background profile. Higher values mean the system quickly forgets past background profiles and prioritizes immediate changes. |
| Histogram Bins | Color Mapping | `[180, 256]` | The resolution of the Hue and Saturation tracking dimensions. Controls how finely granulated the color matching algorithm operates. |
| Histogram Ranges | Color Mapping | `[0, 180, 0, 256]` | The strict coordinate bounds of the HSV channel array. Tells OpenCV exactly where to look within the color spectrum for signature tracking. |
| Mask Penalty Exclusions | Suppression Map | `0` | Forces the target area bounding box pixels to be completely ignored so the system doesn't accidentally learn to suppress the object it is trying to track. |

---

### Predictive Kalman Filter Physics Parameters

| Parameter | Type / Location | Default Value | What It Controls / How to Tune |
| --- | --- | --- | --- |
| `self.kf = cv2.KalmanFilter(4, 2, 0)` | State Dimensions | `4, 2, 0` | Configures the filter matrix arrays. It allocates **4 state variables** ($x, y, dx, dy$), **2 measurement parameters** ($x, y$), and **0 control inputs**. |
| `self.kf.transitionMatrix` | Kinematic Model | Identity Matrix Map | The mathematical equation defining constant velocity physics. It projects the target's future coordinates based on its current position plus its calculated speed. |
| `self.kf.processNoiseCov` | Covariance | `np.eye(4) * 0.03` | **The system's trust in physics laws.** Lower values tell the filter to rigidly trust its velocity calculations (smooth tracking). Higher values make it expect sudden, erratic changes. |
| `self.kf.measurementNoiseCov` | Covariance | `np.eye(2) * 0.4` | **The system's trust in the camera sensor.** Lower values tell the filter to trust raw visual data completely. Higher values tell it to ignore sudden camera shakes and sensor noise. |

---

### Hardware, Frame, & UI Settings

| Parameter | Type / Location | Default Value | What It Controls / How to Tune |
| --- | --- | --- | --- |
| `cv2.CAP_V4L2` | Backend Driver | Backend Toggle | Forces OpenCV to leverage the Linux V4L2 video architecture directly. Crucial on the Jetson Orin Nano to minimize camera capture latency. |
| Frame Width / Height | Frame Configuration | `640` / `480` | Sets the resolution of the video stream. Balanced for fast processing on low-power devices. |
| `cv2.flip(frame, 1)` | Image Processing | `1` | Flips the camera matrix horizontally. Mirrors the display window to make intuitive sense for a desktop environment. |
| `cv2.addWeighted` Gamma | Overlay Mixing | `0.30` | The opacity of the green target mask layer. `0.30` means 30% green overlay and 70% raw background image transparency. |
| Telemetry Print Threshold | Console Output | `2` | Prevents terminal flooding. The console will only log tracking updates if the target moves by more than 2 pixels in either direction. |
