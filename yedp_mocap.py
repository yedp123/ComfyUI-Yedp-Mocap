import os
import torch
import numpy as np
import folder_paths
import cv2
import json
import math

OUTPUT_DIR = folder_paths.get_input_directory()

# --- CONNECTIONS & COLORS (CORRECTED OPENPOSE STANDARD) ---
# OpenCV uses BGR (Blue, Green, Red).
# We map MediaPipe landmarks to OpenPose colors.

POSE_CONNECTIONS = [
    (11, 12), # 0: Shoulders
    (11, 13), # 1: Left Arm Upper (Shoulder->Elbow)
    (13, 15), # 2: Left Arm Lower (Elbow->Wrist)
    (12, 14), # 3: Right Arm Upper (Shoulder->Elbow)
    (14, 16), # 4: Right Arm Lower (Elbow->Wrist)
    (11, 23), # 5: Left Torso (Shoulder->Hip)
    (12, 24), # 6: Right Torso (Shoulder->Hip)
    (23, 24), # 7: Hips
    (23, 25), # 8: Left Leg Upper (Hip->Knee)
    (24, 26), # 9: Right Leg Upper (Hip->Knee)
    (25, 27), # 10: Left Leg Lower (Knee->Ankle)
    (26, 28), # 11: Right Leg Lower (Knee->Ankle)
    (27, 29), # 12: Left Foot (Heel->Toe)
    (28, 30), # 13: Right Foot (Heel->Toe)
    (29, 31), # 14: Left Toe Tip
    (30, 32)  # 15: Right Toe Tip
]

# COLORS (BGR Format)
# MediaPipe "Left" (Odds) = Viewer's Right. Target: Green Arm, Blue Body.
# MediaPipe "Right" (Evens) = Viewer's Left. Target: Orange Arm, Green Body.

POSE_COLORS = [
    (153, 0, 51),    # 0: Shoulders (Dark Purple/Red)
    (0, 255, 0),     # 1: Left Arm Upper (Green)
    (0, 255, 0),     # 2: Left Arm Lower (Green)
    (0, 170, 255),   # 3: Right Arm Upper (Orange)
    (0, 255, 255),   # 4: Right Arm Lower (Yellow)
    (255, 0, 0),     # 5: Left Torso (Blue) - CHANGED to match Left Leg
    (0, 255, 0),     # 6: Right Torso (Green) - CHANGED to match Right Leg
    (153, 0, 51),    # 7: Hips (Dark Purple/Red)
    (255, 0, 0),     # 8: Left Leg Upper (Blue)
    (0, 255, 0),     # 9: Right Leg Upper (Green) - FIXED (Was Cyan)
    (255, 0, 0),     # 10: Left Leg Lower (Blue)
    (0, 255, 0),     # 11: Right Leg Lower (Green) - FIXED
    (255, 0, 0),     # 12: Left Foot (Blue)
    (0, 255, 0),     # 13: Right Foot (Green)
    (255, 0, 0),     # 14: Left Toe (Blue)
    (0, 255, 0)      # 15: Right Toe (Green)
]

# Hand Joint Colors (Standard Rainbow)
HAND_COLORS = [
    (0, 0, 255),       # Wrist (Red)
    (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255),       # Thumb (Red)
    (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255), # Index (Yellow)
    (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),       # Middle (Green)
    (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),       # Ring (Blue)
    (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255)# Pinky (Purple)
]

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]

FACE_INDICES = {
    "oval": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    "lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78],
    "left_eye": [263, 249, 390, 373, 374, 380, 381, 382, 362, 263, 466, 388, 387, 386, 385, 384, 398, 362],
    "left_eyebrow": [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
    "left_iris": [468, 469, 470, 471, 472], 
    "right_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 33, 246, 161, 160, 159, 158, 157, 173, 133],
    "right_eyebrow": [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
    "right_iris": [473, 474, 475, 476, 477], 
    "nose_bridge": [168, 6, 197, 195, 5],
    "nose_bottom": [98, 97, 2, 326, 327],
    "nose_tip": [5, 2] 
}
TORSO_INDICES = [11, 12, 24, 23]

# --- 1 EURO FILTER ---
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = self.dx_prev = self.t_prev = None

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter(self, x, t):
        if self.t_prev is None:
            self.x_prev = x; self.dx_prev = 0; self.t_prev = t
            return x
        t_e = t - self.t_prev
        if t_e <= 0: return self.x_prev
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        self.x_prev = x_hat; self.dx_prev = dx_hat; self.t_prev = t
        return x_hat

# --- BASE CLASS ---
class YedpMocapBase:
    CATEGORY = "Yedp/MoCap"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "POSE_DATA")
    RETURN_NAMES = ("image", "rig_image", "mask", "pose_json")
    FUNCTION = "load_captured_data"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_filename": ("STRING", {"default": "", "multiline": False}),
                "smoothing": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    def apply_one_euro(self, landmarks, filters, timestamp):
        smoothed = []
        if len(filters) != len(landmarks):
            filters.clear()
            for _ in landmarks:
                filters.append({'x': OneEuroFilter(0.1, 0.05), 'y': OneEuroFilter(0.1, 0.05), 'z': OneEuroFilter(0.1, 0.05)})
        
        for i, point in enumerate(landmarks):
            f = filters[i]
            t_sec = timestamp / 1000.0
            new_point = point.copy()
            new_point['x'] = f['x'].filter(point['x'], t_sec)
            new_point['y'] = f['y'].filter(point['y'], t_sec)
            new_point['z'] = f['z'].filter(point['z'], t_sec)
            smoothed.append(new_point)
        return smoothed

    def draw_connections(self, img, landmarks, connections, colors=None, default_color=(0, 0, 255), thickness=2):
        h, w = img.shape[:2]
        for idx, (i, j) in enumerate(connections):
            if i < len(landmarks) and j < len(landmarks):
                p1 = landmarks[i]; p2 = landmarks[j]
                if colors and idx < len(colors):
                    color = colors[idx]
                else:
                    color = default_color
                
                x1, y1 = int(p1['x'] * w), int(p1['y'] * h)
                x2, y2 = int(p2['x'] * w), int(p2['y'] * h)
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    def draw_points(self, img, landmarks, color=(0, 255, 255), radius=3, skip_indices=None):
        h, w = img.shape[:2]
        indices_to_draw = range(len(landmarks))
        
        for i in indices_to_draw:
            if skip_indices and i in skip_indices: continue
            
            p = landmarks[i]
            cx, cy = int(p['x'] * w), int(p['y'] * h)
            
            pt_color = color
            if isinstance(color, list):
                if i < len(color):
                    pt_color = color[i]
                else:
                    pt_color = (255, 255, 255)
            
            cv2.circle(img, (cx, cy), radius, pt_color, -1)

    def draw_subset_points(self, img, landmarks, indices, color=(255, 255, 255), radius=2):
        h, w = img.shape[:2]
        for idx in indices:
            if idx < len(landmarks):
                p = landmarks[idx]
                cx, cy = int(p['x'] * w), int(p['y'] * h)
                cv2.circle(img, (cx, cy), radius, color, -1)

    def draw_contour(self, img, landmarks, indices, color=(255, 255, 255), thickness=1, fill=False):
        h, w = img.shape[:2]
        points = []
        for idx in indices:
            if idx < len(landmarks): points.append([int(landmarks[idx]['x'] * w), int(landmarks[idx]['y'] * h)])
        if len(points) > 0:
            pts = np.array([points], np.int32)
            if fill: cv2.fillPoly(img, pts, color)
            else: cv2.polylines(img, pts, True, color, thickness)

    def fill_convex_hull(self, img, landmarks, color=255):
        h, w = img.shape[:2]
        for i, j in HAND_CONNECTIONS:
            if i < len(landmarks) and j < len(landmarks):
                p1 = landmarks[i]
                p2 = landmarks[j]
                x1, y1 = int(p1['x'] * w), int(p1['y'] * h)
                x2, y2 = int(p2['x'] * w), int(p2['y'] * h)
                cv2.line(img, (x1, y1), (x2, y2), color, 15)
        for p in landmarks:
            cx, cy = int(p['x'] * w), int(p['y'] * h)
            cv2.circle(img, (cx, cy), 8, color, -1)

    def load_captured_data(self, video_filename, smoothing):
        video_path = os.path.join(OUTPUT_DIR, video_filename)
        base_name = os.path.splitext(video_filename)[0]
        json_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")

        frames = []
        if not os.path.exists(video_path):
            return (torch.zeros((1, 512, 512, 3)), torch.zeros((1, 512, 512, 3)), torch.zeros((1, 512, 512)), {})

        ext = os.path.splitext(video_filename)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']:
            img = cv2.imread(video_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                frames.append(img)
                width, height = img.shape[1], img.shape[0]
            else:
                width, height = 512, 512
        else:
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            cap.release()

        rig_frames = []
        mask_frames = []
        final_pose_data = []
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f: raw_data = json.load(f)
                if not isinstance(raw_data, list): raw_data = [raw_data]

                oe_filters = {'pose': [], 'face': [], 'hands': []} 
                pose_face_indices = set(range(11)) 

                for i, frame_data in enumerate(raw_data):
                    rig_canvas = np.zeros((height, width, 3), dtype=np.uint8)
                    mask_canvas = np.zeros((height, width), dtype=np.uint8)
                    processed_frame = frame_data.copy()
                    timestamp = frame_data.get("time", 0)

                    if "face" in frame_data:
                        if smoothing > 0: processed_frame["face"] = self.apply_one_euro(frame_data["face"], oe_filters['face'], timestamp)
                        for key, idxs in FACE_INDICES.items():
                            if key == "oval": 
                                self.draw_contour(mask_canvas, processed_frame["face"], idxs, color=255, fill=True)
                            self.draw_subset_points(rig_canvas, processed_frame["face"], idxs, color=(255, 255, 255), radius=2)

                    if "pose" in frame_data:
                        if smoothing > 0: processed_frame["pose"] = self.apply_one_euro(frame_data["pose"], oe_filters['pose'], timestamp)
                        
                        self.draw_connections(rig_canvas, processed_frame["pose"], POSE_CONNECTIONS, colors=POSE_COLORS, thickness=3)
                        self.draw_points(rig_canvas, processed_frame["pose"], color=(0, 0, 255), radius=4, skip_indices=pose_face_indices)
                        
                        self.draw_contour(mask_canvas, processed_frame["pose"], TORSO_INDICES, color=255, fill=True)
                        self.draw_connections(mask_canvas, processed_frame["pose"], POSE_CONNECTIONS, default_color=255, thickness=80) 
                        self.draw_points(mask_canvas, processed_frame["pose"], color=255, radius=20, skip_indices=pose_face_indices)

                    if "hands" in frame_data:
                        for hand_landmarks in frame_data["hands"]:
                            self.draw_connections(rig_canvas, hand_landmarks, HAND_CONNECTIONS, default_color=(50, 50, 50), thickness=2)
                            self.draw_points(rig_canvas, hand_landmarks, color=HAND_COLORS, radius=4)
                            self.fill_convex_hull(mask_canvas, hand_landmarks, color=255)

                    rig_canvas = cv2.cvtColor(rig_canvas, cv2.COLOR_BGR2RGB)
                    
                    rig_frames.append(rig_canvas.astype(np.float32) / 255.0)
                    mask_frames.append(mask_canvas.astype(np.float32) / 255.0)
                    final_pose_data.append(processed_frame)
            except Exception as e: print(f"[Yedp] Error: {e}")

        while len(rig_frames) < len(frames):
            rig_frames.append(np.zeros((height, width, 3), dtype=np.float32))
            mask_frames.append(np.zeros((height, width), dtype=np.float32))

        if not frames: 
            return (torch.zeros((1, 512, 512, 3)), torch.zeros((1, 512, 512, 3)), torch.zeros((1, 512, 512)), {})

        return (torch.from_numpy(np.array(frames)), torch.from_numpy(np.array(rig_frames)), torch.from_numpy(np.array(mask_frames)), final_pose_data)

# --- NODE DEFINITIONS ---
class YedpWebcamRecorder(YedpMocapBase):
    pass

class YedpWebcamSnapshot(YedpMocapBase):
    @classmethod
    def INPUT_TYPES(s):
        types = YedpMocapBase.INPUT_TYPES()
        types["required"]["video_filename"] = ("STRING", {"default": "snapshot.png", "multiline": False})
        return types

class YedpVideoMoCap(YedpMocapBase):
    @classmethod
    def INPUT_TYPES(s):
        types = YedpMocapBase.INPUT_TYPES()
        return types

class YedpImageMoCap(YedpMocapBase):
    @classmethod
    def INPUT_TYPES(s):
        types = YedpMocapBase.INPUT_TYPES()
        types["required"]["video_filename"] = ("STRING", {"default": "image.png", "multiline": False})
        return types

NODE_CLASS_MAPPINGS = {
    "YedpWebcamRecorder": YedpWebcamRecorder,
    "YedpWebcamSnapshot": YedpWebcamSnapshot,
    "YedpVideoMoCap": YedpVideoMoCap,
    "YedpImageMoCap": YedpImageMoCap
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YedpWebcamRecorder": "Yedp Webcam Recorder (Video)",
    "YedpWebcamSnapshot": "Yedp Webcam Snapshot (Image)",
    "YedpVideoMoCap": "Yedp Video MoCap (File)",
    "YedpImageMoCap": "Yedp Image MoCap (File)"
}
