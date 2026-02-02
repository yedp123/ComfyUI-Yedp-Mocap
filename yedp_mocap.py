import os
import torch
import numpy as np
import folder_paths
import cv2
import json
import math
import random

OUTPUT_DIR = folder_paths.get_input_directory()
TEMP_DIR = folder_paths.get_temp_directory()

# ==============================================================================
# 1. USER CONFIGURABLE COLORS (RGB Format)
# ==============================================================================
JOINT_COLORS = {
    "Nose": (255, 0, 0),
    "Neck": (255, 85, 0),
    "R_Shoulder": (255, 170, 0),
    "R_Elbow": (255, 255, 0),
    "R_Wrist": (170, 255, 0),
    "L_Shoulder": (85, 255, 0),
    "L_Elbow": (0, 255, 0),
    "L_Wrist": (0, 255, 85),
    "R_Hip": (0, 255, 170),
    "R_Knee": (0, 255, 255),
    "R_Ankle": (0, 170, 255),
    "L_Hip": (0, 85, 255),
    "L_Knee": (0, 0, 255),
    "L_Ankle": (85, 0, 255),
    "R_Eye": (170, 0, 255),
    "L_Eye": (255, 0, 255),
    "R_Ear": (255, 0, 170),
    "L_Ear": (255, 0, 85),
}

BONE_COLORS = {
    "R_Shoulderblade": (153, 0, 0), "L_Shoulderblade": (153, 51, 0),
    "R_Arm": (153, 102, 0), "R_Forearm": (153, 153, 0),
    "L_Arm": (102, 153, 0), "L_Forearm": (51, 153, 0),
    "R_Torso": (0, 153, 0), "R_Thigh": (0, 153, 51), "R_Calf": (0, 153, 102),
    "L_Torso": (0, 153, 153), "L_Thigh": (0, 102, 153), "L_Calf": (0, 51, 153),
    "Head": (0, 0, 153),
    "R_Eyebrow": (51, 0, 153), "R_EarLine": (102, 0, 153),
    "L_Eyebrow": (153, 0, 153), "L_EarLine": (153, 0, 102),
}

def get_bgr(color_dict, key):
    c = color_dict.get(key, (255, 255, 255))
    return (c[2], c[1], c[0])

HAND_BONE_COLORS = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]

# ==============================================================================
# 2. MAPPINGS
# ==============================================================================
MP_TO_JOINT = {
    0: "Nose", 12: "R_Shoulder", 14: "R_Elbow", 16: "R_Wrist",
    11: "L_Shoulder", 13: "L_Elbow", 15: "L_Wrist",
    24: "R_Hip", 26: "R_Knee", 28: "R_Ankle",
    23: "L_Hip", 25: "L_Knee", 27: "L_Ankle",
    5: "R_Eye", 2: "L_Eye", 8: "R_Ear", 7: "L_Ear"
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

# EXACT 70-POINT OPENPOSE FACE DEFINITION
FACE_INDICES_SPARSE = {
    # Jaw: 17 Points (Right Ear -> Chin -> Left Ear). No Forehead.
    "jaw": [234, 93, 132, 58, 172, 136, 150, 176, 152, 400, 379, 365, 397, 288, 361, 323, 454],
    
    # Eyebrows: 5 points each (Top line only)
    "right_eyebrow": [46, 53, 52, 65, 55],
    "left_eyebrow": [276, 283, 282, 295, 285],
    
    # Nose: 4 Bridge + 5 Bottom = 9 points
    "nose_bridge": [6, 197, 195, 5], 
    "nose_bottom": [98, 97, 2, 326, 327],
    
    # Eyes: 6 points each
    "right_eye": [33, 160, 158, 133, 153, 144], 
    "left_eye": [362, 385, 387, 263, 373, 380], 
    
    # Outer Lips: 12 Points (Manually selected to match the 12-point standard)
    "lips_outer": [61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91],
    
    # Inner Lips: 8 Points (Manually selected to match the 8-point standard)
    "lips_inner": [78, 80, 13, 311, 308, 402, 14, 178]
}
TORSO_INDICES = [11, 12, 24, 23] 

# --- UTILS ---
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff; self.beta = beta; self.d_cutoff = d_cutoff; self.x_prev = self.dx_prev = self.t_prev = None
    def smoothing_factor(self, t_e, cutoff): return (2 * math.pi * cutoff * t_e) / ((2 * math.pi * cutoff * t_e) + 1)
    def exponential_smoothing(self, a, x, x_prev): return a * x + (1 - a) * x_prev
    def filter(self, x, t):
        if self.t_prev is None: self.x_prev = x; self.dx_prev = 0; self.t_prev = t; return x
        t_e = t - self.t_prev; 
        if t_e <= 0: return self.x_prev
        a_d = self.smoothing_factor(t_e, self.d_cutoff); dx = (x - self.x_prev) / t_e; dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat); a = self.smoothing_factor(t_e, cutoff); x_hat = self.exponential_smoothing(a, x, self.x_prev)
        self.x_prev = x_hat; self.dx_prev = dx_hat; self.t_prev = t; return x_hat

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
                "include_dense_face": ("BOOLEAN", {"default": False, "label_on": "Dense (468 pts)", "label_off": "Sparse (Standard)"}),
            },
        }

    def apply_one_euro(self, landmarks, filters, timestamp):
        smoothed = []
        if len(filters) != len(landmarks):
            filters.clear()
            for _ in landmarks: filters.append({'x': OneEuroFilter(0.1, 0.05), 'y': OneEuroFilter(0.1, 0.05), 'z': OneEuroFilter(0.1, 0.05)})
        for i, point in enumerate(landmarks):
            f = filters[i]; t_sec = timestamp / 1000.0; new_point = point.copy()
            new_point['x'] = f['x'].filter(point['x'], t_sec); new_point['y'] = f['y'].filter(point['y'], t_sec); new_point['z'] = f['z'].filter(point['z'], t_sec)
            smoothed.append(new_point)
        return smoothed

    def draw_line(self, img, p1, p2, color, thickness=3, w=512, h=512):
        x1, y1 = int(p1['x'] * w), int(p1['y'] * h)
        x2, y2 = int(p2['x'] * w), int(p2['y'] * h)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

    def draw_point(self, img, p, color, radius=4, w=512, h=512):
        cx, cy = int(p['x'] * w), int(p['y'] * h)
        cv2.circle(img, (cx, cy), radius, color, -1, lineType=cv2.LINE_AA)

    def fill_hand_hull(self, img, landmarks, color=255):
        h, w = img.shape[:2]
        for i, j in HAND_CONNECTIONS:
            self.draw_line(img, landmarks[i], landmarks[j], color, 15, w, h)
        for p in landmarks:
            self.draw_point(img, p, color, 8, w, h)

    def load_captured_data(self, video_filename, smoothing, include_dense_face):
        video_path = os.path.join(OUTPUT_DIR, video_filename)
        base_name = os.path.splitext(video_filename)[0]
        json_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")

        frames = []
        if not os.path.exists(video_path): return (torch.zeros((1, 512, 512, 3)), torch.zeros((1, 512, 512, 3)), torch.zeros((1, 512, 512)), {})
             
        ext = os.path.splitext(video_filename)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']:
            img = cv2.imread(video_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); img = img.astype(np.float32) / 255.0; frames.append(img); width, height = img.shape[1], img.shape[0]
            else: width, height = 512, 512
        else:
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            while cap.isOpened():
                ret, frame = cap.read(); 
                if not ret: break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); frame = frame.astype(np.float32) / 255.0; frames.append(frame)
            cap.release()

        rig_frames = []
        mask_frames = []
        final_pose_data = []
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f: raw_data = json.load(f)
                if not isinstance(raw_data, list): raw_data = [raw_data]
                oe_filters = {'pose': [], 'face': [], 'hands': []} 
                
                for i, frame_data in enumerate(raw_data):
                    rig_canvas = np.zeros((height, width, 3), dtype=np.uint8)
                    mask_canvas = np.zeros((height, width), dtype=np.uint8)
                    processed_frame = frame_data.copy()
                    timestamp = frame_data.get("time", 0)

                    # --- DETECT HANDS ---
                    r_hand_root = None; l_hand_root = None
                    if "pose" in frame_data and "hands" in frame_data:
                        pose = processed_frame["pose"]
                        r_wrist_body = pose[16]; l_wrist_body = pose[15]
                        for hand in frame_data["hands"]:
                            if not hand: continue
                            root = hand[0]
                            dist_r = math.hypot(root['x'] - r_wrist_body['x'], root['y'] - r_wrist_body['y'])
                            dist_l = math.hypot(root['x'] - l_wrist_body['x'], root['y'] - l_wrist_body['y'])
                            if dist_r < dist_l: r_hand_root = root
                            else: l_hand_root = root

                    # --- FACE ---
                    if "face" in frame_data:
                        if smoothing > 0: processed_frame["face"] = self.apply_one_euro(frame_data["face"], oe_filters['face'], timestamp)
                        
                        # 1. RIG DRAWING
                        if include_dense_face:
                            for point in processed_frame["face"]:
                                self.draw_point(rig_canvas, point, (255,255,255), radius=1, w=width, h=height)
                        else:
                            for key, idxs in FACE_INDICES_SPARSE.items():
                                 for idx in idxs:
                                    if idx < len(processed_frame["face"]):
                                        self.draw_point(rig_canvas, processed_frame["face"][idx], (255,255,255), radius=2, w=width, h=height)

                        if 468 < len(processed_frame["face"]):
                            self.draw_point(rig_canvas, processed_frame["face"][468], (0, 255, 255), radius=2, w=width, h=height)
                        if 473 < len(processed_frame["face"]):
                            self.draw_point(rig_canvas, processed_frame["face"][473], (0, 255, 255), radius=2, w=width, h=height)

                        hull_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
                        pts = []
                        for idx in hull_indices: 
                            if idx < len(processed_frame["face"]):
                                pts.append([int(processed_frame["face"][idx]['x'] * width), int(processed_frame["face"][idx]['y'] * height)])
                        if pts: cv2.fillPoly(mask_canvas, np.array([pts], np.int32), 255)

                    # --- BODY ---
                    if "pose" in frame_data:
                        if smoothing > 0: processed_frame["pose"] = self.apply_one_euro(frame_data["pose"], oe_filters['pose'], timestamp)
                        pose = processed_frame["pose"]
                        neck = {'x': (pose[11]['x'] + pose[12]['x']) / 2, 'y': (pose[11]['y'] + pose[12]['y']) / 2}
                        nose = pose[0]
                        
                        self.draw_line(rig_canvas, nose, neck, get_bgr(BONE_COLORS, "Head"), 3, width, height)
                        self.draw_line(rig_canvas, neck, pose[24], get_bgr(BONE_COLORS, "R_Torso"), 3, width, height)
                        self.draw_line(rig_canvas, neck, pose[23], get_bgr(BONE_COLORS, "L_Torso"), 3, width, height)
                        self.draw_line(rig_canvas, neck, pose[12], get_bgr(BONE_COLORS, "R_Shoulderblade"), 3, width, height)
                        self.draw_line(rig_canvas, pose[12], pose[14], get_bgr(BONE_COLORS, "R_Arm"), 3, width, height)
                        r_wrist_end = r_hand_root if r_hand_root else pose[16]
                        self.draw_line(rig_canvas, pose[14], r_wrist_end, get_bgr(BONE_COLORS, "R_Forearm"), 3, width, height)
                        self.draw_line(rig_canvas, neck, pose[11], get_bgr(BONE_COLORS, "L_Shoulderblade"), 3, width, height)
                        self.draw_line(rig_canvas, pose[11], pose[13], get_bgr(BONE_COLORS, "L_Arm"), 3, width, height)
                        l_wrist_end = l_hand_root if l_hand_root else pose[15]
                        self.draw_line(rig_canvas, pose[13], l_wrist_end, get_bgr(BONE_COLORS, "L_Forearm"), 3, width, height)
                        self.draw_line(rig_canvas, pose[24], pose[26], get_bgr(BONE_COLORS, "R_Thigh"), 3, width, height)
                        self.draw_line(rig_canvas, pose[26], pose[28], get_bgr(BONE_COLORS, "R_Calf"), 3, width, height)
                        self.draw_line(rig_canvas, pose[23], pose[25], get_bgr(BONE_COLORS, "L_Thigh"), 3, width, height)
                        self.draw_line(rig_canvas, pose[25], pose[27], get_bgr(BONE_COLORS, "L_Calf"), 3, width, height)

                        if 7 < len(pose) and 8 < len(pose):
                            self.draw_line(rig_canvas, nose, pose[5], get_bgr(BONE_COLORS, "R_Eyebrow"), 3, width, height)
                            self.draw_line(rig_canvas, pose[5], pose[8], get_bgr(BONE_COLORS, "R_EarLine"), 3, width, height)
                            self.draw_line(rig_canvas, nose, pose[2], get_bgr(BONE_COLORS, "L_Eyebrow"), 3, width, height)
                            self.draw_line(rig_canvas, pose[2], pose[7], get_bgr(BONE_COLORS, "L_EarLine"), 3, width, height)

                        self.draw_point(rig_canvas, neck, get_bgr(JOINT_COLORS, "Neck"), 4, width, height)
                        for idx, name in MP_TO_JOINT.items():
                            if idx < len(pose):
                                if name == "R_Wrist" and r_hand_root: continue
                                if name == "L_Wrist" and l_hand_root: continue
                                self.draw_point(rig_canvas, pose[idx], get_bgr(JOINT_COLORS, name), 4, width, height)

                        pts = []; 
                        for idx in TORSO_INDICES: pts.append([int(pose[idx]['x'] * width), int(pose[idx]['y'] * height)])
                        if pts: cv2.fillPoly(mask_canvas, np.array([pts], np.int32), 255)

                    # --- HANDS ---
                    if "hands" in frame_data:
                        for hand in frame_data["hands"]:
                            for i, (start, end) in enumerate(HAND_CONNECTIONS):
                                color = HAND_BONE_COLORS[i // 4]
                                self.draw_line(rig_canvas, hand[start], hand[end], color, 2, width, height)
                            for i, p in enumerate(hand):
                                color_idx = 0 if i < 1 else (i - 1) // 4
                                self.draw_point(rig_canvas, p, HAND_BONE_COLORS[min(color_idx, 4)], 4, width, height)
                            self.fill_hand_hull(mask_canvas, hand, color=255)

                    rig_canvas = cv2.cvtColor(rig_canvas, cv2.COLOR_BGR2RGB)
                    rig_frames.append(rig_canvas.astype(np.float32) / 255.0)
                    mask_frames.append(mask_canvas.astype(np.float32) / 255.0)
                    final_pose_data.append(processed_frame)
            except Exception as e: print(f"[Yedp] Error: {e}")

        while len(rig_frames) < len(frames):
            rig_frames.append(np.zeros((height, width, 3), dtype=np.float32)); mask_frames.append(np.zeros((height, width), dtype=np.float32))

        if not frames: return (torch.zeros((1, 512, 512, 3)), torch.zeros((1, 512, 512, 3)), torch.zeros((1, 512, 512)), {})
        return (torch.from_numpy(np.array(frames)), torch.from_numpy(np.array(rig_frames)), torch.from_numpy(np.array(mask_frames)), final_pose_data)

class YedpWebcamRecorder(YedpMocapBase): pass
class YedpWebcamSnapshot(YedpMocapBase):
    @classmethod
    def INPUT_TYPES(s): types = YedpMocapBase.INPUT_TYPES(); types["required"]["video_filename"] = ("STRING", {"default": "snapshot.png", "multiline": False}); return types
class YedpVideoMoCap(YedpMocapBase):
    @classmethod
    def INPUT_TYPES(s): return YedpMocapBase.INPUT_TYPES()
class YedpImageMoCap(YedpMocapBase):
    @classmethod
    def INPUT_TYPES(s): types = YedpMocapBase.INPUT_TYPES(); types["required"]["video_filename"] = ("STRING", {"default": "image.png", "multiline": False}); return types

# ==============================================================================
# NEW NODE: 3D POSE VIEWER (Three.js)
# ==============================================================================
class Yedp3DViewer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_json": ("POSE_DATA",),
            },
        }

    RETURN_TYPES = ("POSE_DATA",)
    RETURN_NAMES = ("pose_json",)
    FUNCTION = "view_3d"
    CATEGORY = "Yedp/MoCap"
    OUTPUT_NODE = True

    def view_3d(self, pose_json):
        # We pass the full JSON data to the frontend
        # ComfyUI allows passing arbitrary data in the "ui" dictionary
        return { "ui": { "yedp_3d_data": pose_json }, "result": (pose_json,) }


NODE_CLASS_MAPPINGS = { 
    "YedpWebcamRecorder": YedpWebcamRecorder, 
    "YedpWebcamSnapshot": YedpWebcamSnapshot, 
    "YedpVideoMoCap": YedpVideoMoCap, 
    "YedpImageMoCap": YedpImageMoCap,
    "Yedp3DViewer": Yedp3DViewer
}

NODE_DISPLAY_NAME_MAPPINGS = { 
    "YedpWebcamRecorder": "Yedp Webcam Recorder (Video)", 
    "YedpWebcamSnapshot": "Yedp Webcam Snapshot (Image)", 
    "YedpVideoMoCap": "Yedp Video MoCap (File)", 
    "YedpImageMoCap": "Yedp Image MoCap (File)",
    "Yedp3DViewer": "Yedp 3D Viewer"
}
