import os
import torch
import numpy as np
import folder_paths
import cv2
import json
import math
import torchaudio

# Define the directory to save captured videos
OUTPUT_DIR = folder_paths.get_input_directory()

# --- CONNECTIONS & INDICES ---
POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32)]
POSE_COLORS = [(255, 0, 85), (255, 255, 0), (255, 170, 0), (0, 255, 0), (0, 170, 255), (255, 0, 0), (255, 0, 170), (255, 0, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85), (255, 0, 0)]
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
# Mask Volume Indices
TORSO_INDICES = [11, 12, 24, 23]
PALM_INDICES = [0, 1, 5, 9, 13, 17]

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
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "POSE_DATA", "AUDIO")
    RETURN_NAMES = ("image", "rig_image", "mask", "pose_json", "audio")
    FUNCTION = "load_captured_data"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_filename": ("STRING", {"default": "", "multiline": False}), # Hidden input populated by JS
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

    def draw_connections(self, img, landmarks, connections, colors=None, default_color=(255, 0, 0), thickness=2):
        h, w = img.shape[:2]
        for idx, (i, j) in enumerate(connections):
            if i < len(landmarks) and j < len(landmarks):
                p1 = landmarks[i]; p2 = landmarks[j]
                color = colors[idx % len(colors)] if colors else default_color
                x1, y1 = int(p1['x'] * w), int(p1['y'] * h)
                x2, y2 = int(p2['x'] * w), int(p2['y'] * h)
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    def draw_points(self, img, landmarks, color=(0, 255, 255), radius=3, skip_indices=None):
        h, w = img.shape[:2]
        for i, p in enumerate(landmarks):
            if skip_indices and i in skip_indices: continue
            cx, cy = int(p['x'] * w), int(p['y'] * h)
            pt_color = color if len(img.shape) == 3 else 255
            cv2.circle(img, (cx, cy), radius, pt_color, -1)

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
        # IMPROVED MASK: Replaces 'mitten' convex hull with thick lines for fingers
        h, w = img.shape[:2]
        # Draw thick lines for finger segments
        for i, j in HAND_CONNECTIONS:
            if i < len(landmarks) and j < len(landmarks):
                p1 = landmarks[i]
                p2 = landmarks[j]
                x1, y1 = int(p1['x'] * w), int(p1['y'] * h)
                x2, y2 = int(p2['x'] * w), int(p2['y'] * h)
                # Thickness ~15-20px fills out the finger volume nicely
                cv2.line(img, (x1, y1), (x2, y2), color, 15)
        
        # Draw round joints to smooth connections
        for p in landmarks:
            cx, cy = int(p['x'] * w), int(p['y'] * h)
            cv2.circle(img, (cx, cy), 8, color, -1)

    def load_audio(self, file_path):
        # FIX: Robust audio loading. If missing or fails, return 1 sec silence.
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        except Exception as e:
            # Create a silent dummy audio to prevent nodes from crashing
            # 1 channel, 44100 Hz, 1 second
            silent = torch.zeros((1, 44100))
            return {"waveform": silent.unsqueeze(0), "sample_rate": 44100}

    def load_captured_data(self, video_filename, smoothing):
        video_path = os.path.join(OUTPUT_DIR, video_filename)
        base_name = os.path.splitext(video_filename)[0]
        json_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")

        # 1. Load Media (Video or Image)
        frames = []
        audio = None
        
        if not os.path.exists(video_path):
            # Return empty structure with silent audio
            silent = self.load_audio("") 
            return (torch.zeros((1, 512, 512, 3)), torch.zeros((1, 512, 512, 3)), torch.zeros((1, 512, 512)), {}, silent)

        # Check extension for Image vs Video
        ext = os.path.splitext(video_filename)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']:
            # Image Mode
            img = cv2.imread(video_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                frames.append(img)
                width, height = img.shape[1], img.shape[0]
            else:
                width, height = 512, 512
            # Audio for image is silence
            audio = self.load_audio("")
        else:
            # Video Mode
            audio = self.load_audio(video_path)
            
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

        # DYNAMIC THICKNESS CALCULATION (Auto-Scaling)
        # Scale based on 720p baseline. If video is 4K, lines will be ~3x thicker.
        scale_factor = min(width, height) / 720.0
        line_thick = max(2, int(3 * scale_factor))
        dot_radius = max(3, int(4 * scale_factor))
        mask_thick = max(15, int(80 * scale_factor)) # For volumetric mask

        # 2. Process JSON
        rig_frames = []
        mask_frames = []
        final_pose_data = []
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f: raw_data = json.load(f)
                
                # If single object (Snapshot), wrap in list
                if not isinstance(raw_data, list): raw_data = [raw_data]

                oe_filters = {'pose': [], 'face': [], 'hands': []} 
                pose_face_indices = set(range(11)) 

                for i, frame_data in enumerate(raw_data):
                    rig_canvas = np.zeros((height, width, 3), dtype=np.uint8)
                    mask_canvas = np.zeros((height, width), dtype=np.uint8)
                    processed_frame = frame_data.copy()
                    timestamp = frame_data.get("time", 0)

                    # FACE
                    if "face" in frame_data:
                        if smoothing > 0: processed_frame["face"] = self.apply_one_euro(frame_data["face"], oe_filters['face'], timestamp)
                        for key, idxs in FACE_INDICES.items():
                            if "iris" in key: self.draw_contour(rig_canvas, processed_frame["face"], idxs, color=(255, 0, 0), thickness=1, fill=True)
                            else: self.draw_contour(rig_canvas, processed_frame["face"], idxs, color=(255, 255, 255), thickness=max(1, int(1*scale_factor)), fill=False)
                            if key == "oval": self.draw_contour(mask_canvas, processed_frame["face"], idxs, color=255, fill=True)

                    # POSE
                    if "pose" in frame_data:
                        if smoothing > 0: processed_frame["pose"] = self.apply_one_euro(frame_data["pose"], oe_filters['pose'], timestamp)
                        self.draw_connections(rig_canvas, processed_frame["pose"], POSE_CONNECTIONS, colors=POSE_COLORS, thickness=line_thick)
                        self.draw_points(rig_canvas, processed_frame["pose"], color=(0, 0, 255), radius=dot_radius, skip_indices=pose_face_indices)
                        # Mask (Volume)
                        self.draw_contour(mask_canvas, processed_frame["pose"], TORSO_INDICES, color=255, fill=True)
                        self.draw_connections(mask_canvas, processed_frame["pose"], POSE_CONNECTIONS, default_color=255, thickness=mask_thick) 
                        self.draw_points(mask_canvas, processed_frame["pose"], color=255, radius=int(mask_thick/2), skip_indices=pose_face_indices)

                    # HANDS
                    if "hands" in frame_data:
                        for hand_landmarks in frame_data["hands"]:
                            self.draw_connections(rig_canvas, hand_landmarks, HAND_CONNECTIONS, default_color=(200, 200, 200), thickness=max(1, int(2*scale_factor)))
                            self.draw_points(rig_canvas, hand_landmarks, color=(0, 255, 0), radius=max(2, int(3*scale_factor)))
                            
                            # Use new fill logic for mask
                            self.fill_convex_hull(mask_canvas, hand_landmarks, color=255)

                    rig_frames.append(rig_canvas.astype(np.float32) / 255.0)
                    mask_frames.append(mask_canvas.astype(np.float32) / 255.0)
                    final_pose_data.append(processed_frame)
            except Exception as e: print(f"[Yedp] Error: {e}")

        # Padding / Sync
        while len(rig_frames) < len(frames):
            rig_frames.append(np.zeros((height, width, 3), dtype=np.float32))
            mask_frames.append(np.zeros((height, width), dtype=np.float32))

        # Handle case where no JSON or empty frames
        if not frames:
            silent = self.load_audio("") 
            return (torch.zeros((1, 512, 512, 3)), torch.zeros((1, 512, 512, 3)), torch.zeros((1, 512, 512)), {}, silent)

        return (torch.from_numpy(np.array(frames)), torch.from_numpy(np.array(rig_frames)), torch.from_numpy(np.array(mask_frames)), final_pose_data, audio)

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
        # Allows user to pick a video for processing if they want to override the UI drag-drop
        # But mostly controlled by JS
        return types

class YedpImageMoCap(YedpMocapBase):
    @classmethod
    def INPUT_TYPES(s):
        types = YedpMocapBase.INPUT_TYPES()
        # For loading existing image files that might have corresponding JSON data
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
