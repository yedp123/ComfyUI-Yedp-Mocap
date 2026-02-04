import os
import torch
import numpy as np
import folder_paths
import cv2
import json
import math
import copy

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

# Default neutral color (fallback)
WRIST_ROOT_COLOR = (168, 168, 168) 

# Gradient colors for hands (used for LINES/BONES only): [Wrist/Root, Joint1, Joint2, Joint3, Tip]
HAND_GRADIENT_COLORS = [
    [WRIST_ROOT_COLOR, (100, 0, 0),   (150, 0, 0),   (200, 0, 0),   (255, 0, 0)],   # Thumb
    [WRIST_ROOT_COLOR, (100, 100, 0), (150, 150, 0), (200, 200, 0), (255, 255, 0)], # Index
    [WRIST_ROOT_COLOR, (0, 100, 50),  (0, 150, 75),  (0, 200, 100), (0, 255, 125)], # Middle
    [WRIST_ROOT_COLOR, (0, 50, 100),  (0, 75, 150),  (0, 100, 200), (0, 125, 255)], # Ring
    [WRIST_ROOT_COLOR, (100, 0, 100), (150, 0, 150), (200, 0, 200), (255, 0, 255)]  # Pinky
]

def get_bgr(color_dict, key):
    c = color_dict.get(key, (255, 255, 255))
    return (c[2], c[1], c[0])

def get_hand_bgr(finger_idx, joint_idx):
    c = HAND_GRADIENT_COLORS[finger_idx][joint_idx]
    return (c[2], c[1], c[0])

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
    (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8), # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

FACE_INDICES_SPARSE = {
    "jaw": [234, 93, 132, 58, 172, 136, 150, 176, 152, 400, 379, 365, 397, 288, 361, 323, 454],
    "right_eyebrow": [46, 53, 52, 65, 55], "left_eyebrow": [276, 283, 282, 295, 285],
    "nose_bridge": [6, 197, 195, 5], "nose_bottom": [98, 97, 2, 326, 327],
    "right_eye": [33, 160, 158, 133, 153, 144], "left_eye": [362, 385, 387, 263, 373, 380], 
    "lips_outer": [61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91],
    "lips_inner": [78, 80, 13, 311, 308, 402, 14, 178]
}

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
                "include_dense_face": ("BOOLEAN", {"default": False}),
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
        for i, j in HAND_CONNECTIONS: self.draw_line(img, landmarks[i], landmarks[j], color, 12, w, h)
        for p in landmarks: self.draw_point(img, p, color, 8, w, h)

    def load_captured_data(self, video_filename, smoothing, include_dense_face):
        video_path = os.path.join(OUTPUT_DIR, video_filename)
        base_name = os.path.splitext(video_filename)[0]
        json_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")

        frames = []
        if not os.path.exists(video_path):
            return (torch.zeros((1, 512, 512, 3)), torch.zeros((1, 512, 512, 3)), torch.zeros((1, 512, 512)), [])
              
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            img = cv2.imread(video_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img.astype(np.float32) / 255.0)
                height, width = img.shape[:2]
            else:
                return (torch.zeros((1, 512, 512, 3)), torch.zeros((1, 512, 512, 3)), torch.zeros((1, 512, 512)), [])
        else:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            while cap.isOpened():
                ret, frame = cap.read() 
                if not ret: break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame.astype(np.float32) / 255.0)
            cap.release()

        rig_frames = []; mask_frames = []; final_pose_data = []
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f: raw_data = json.load(f)
                if not isinstance(raw_data, list): raw_data = [raw_data]
                
                oe_filters = {'pose': [], 'face': [], 'hands': []} 
                
                for i in range(len(frames)):
                    data_idx = i if i < len(raw_data) else len(raw_data) - 1
                    frame_data = raw_data[data_idx]
                    
                    rig_canvas = np.zeros((height, width, 3), dtype=np.uint8)
                    mask_canvas = np.zeros((height, width), dtype=np.uint8)
                    
                    base_frame = copy.deepcopy(frame_data)
                    timestamp = frame_data.get("time", 0)

                    if smoothing > 0:
                        if "face" in base_frame and base_frame["face"]:
                            base_frame["face"] = self.apply_one_euro(base_frame["face"], oe_filters['face'], timestamp)
                        if "pose" in base_frame and base_frame["pose"]:
                            base_frame["pose"] = self.apply_one_euro(base_frame["pose"], oe_filters['pose'], timestamp)

                    frame_2d = copy.deepcopy(base_frame)
                    
                    # --- Body & Head Drawing ---
                    if "pose" in frame_2d and frame_2d["pose"]:
                        pose = frame_2d["pose"]
                        neck = {'x': (pose[11]['x'] + pose[12]['x']) / 2, 'y': (pose[11]['y'] + pose[12]['y']) / 2}
                        
                        # Head Bone Rig
                        self.draw_line(rig_canvas, pose[0], neck, get_bgr(BONE_COLORS, "Head"), 3, width, height)
                        self.draw_line(rig_canvas, pose[0], pose[5], get_bgr(BONE_COLORS, "R_Eyebrow"), 3, width, height)
                        self.draw_line(rig_canvas, pose[5], pose[8], get_bgr(BONE_COLORS, "R_EarLine"), 3, width, height)
                        self.draw_line(rig_canvas, pose[0], pose[2], get_bgr(BONE_COLORS, "L_Eyebrow"), 3, width, height)
                        self.draw_line(rig_canvas, pose[2], pose[7], get_bgr(BONE_COLORS, "L_EarLine"), 3, width, height)
                        
                        # Head Mask (Ellipse / Oval)
                        ear_dist = math.hypot(pose[8]['x'] - pose[7]['x'], pose[8]['y'] - pose[7]['y'])
                        hr = int((ear_dist * width) * 0.8) if ear_dist > 0 else 40
                        cv2.ellipse(mask_canvas, (int(pose[0]['x']*width), int(pose[0]['y']*height)), (hr, int(hr*1.2)), 0, 0, 360, 255, -1, lineType=cv2.LINE_AA)
                        self.draw_line(mask_canvas, pose[0], neck, 255, 20, width, height)

                        # Torso Rig & Mask
                        self.draw_line(rig_canvas, neck, pose[24], get_bgr(BONE_COLORS, "R_Torso"), 3, width, height)
                        self.draw_line(rig_canvas, neck, pose[23], get_bgr(BONE_COLORS, "L_Torso"), 3, width, height)
                        self.draw_line(rig_canvas, neck, pose[12], get_bgr(BONE_COLORS, "R_Shoulderblade"), 3, width, height)
                        self.draw_line(rig_canvas, pose[12], pose[14], get_bgr(BONE_COLORS, "R_Arm"), 3, width, height)
                        self.draw_line(rig_canvas, pose[14], pose[16], get_bgr(BONE_COLORS, "R_Forearm"), 3, width, height)
                        self.draw_line(rig_canvas, neck, pose[11], get_bgr(BONE_COLORS, "L_Shoulderblade"), 3, width, height)
                        self.draw_line(rig_canvas, pose[11], pose[13], get_bgr(BONE_COLORS, "L_Arm"), 3, width, height)
                        self.draw_line(rig_canvas, pose[13], pose[15], get_bgr(BONE_COLORS, "L_Forearm"), 3, width, height)
                        
                        torso_poly = np.array([
                            [int(pose[12]['x'] * width), int(pose[12]['y'] * height)],
                            [int(pose[11]['x'] * width), int(pose[11]['y'] * height)],
                            [int(pose[23]['x'] * width), int(pose[23]['y'] * height)],
                            [int(pose[24]['x'] * width), int(pose[24]['y'] * height)]
                        ], dtype=np.int32)
                        cv2.fillPoly(mask_canvas, [torso_poly], 255, lineType=cv2.LINE_AA)
                        
                        for p1, p2 in [(pose[12], pose[14]), (pose[14], pose[16]), (pose[11], pose[13]), (pose[13], pose[15])]:
                            self.draw_line(mask_canvas, p1, p2, 255, 20, width, height)

                        # Legs
                        for p1, p2, color_key in [(pose[24], pose[26], "R_Thigh"), (pose[26], pose[28], "R_Calf"), (pose[23], pose[25], "L_Thigh"), (pose[25], pose[27], "L_Calf")]:
                            self.draw_line(rig_canvas, p1, p2, get_bgr(BONE_COLORS, color_key), 3, width, height)
                            self.draw_line(mask_canvas, p1, p2, 255, 25, width, height)

                        self.draw_point(rig_canvas, neck, get_bgr(JOINT_COLORS, "Neck"), 4, width, height)
                        for idx, name in MP_TO_JOINT.items():
                            if idx < len(pose): self.draw_point(rig_canvas, pose[idx], get_bgr(JOINT_COLORS, name), 4, width, height)
                    
                    # --- Face Mask/Rig (Independent of Torso) ---
                    if "face" in frame_2d and frame_2d["face"]:
                        face = frame_2d["face"]
                        
                        # Head Oval Mask fallback if no pose data
                        if not ("pose" in frame_2d and frame_2d["pose"]):
                            nose = face[1] # Approximate nose center
                            ear_l, ear_r = face[234], face[454]
                            ear_dist = math.hypot(ear_l['x']-ear_r['x'], ear_l['y']-ear_r['y'])
                            hr = int((ear_dist * width) * 0.6) if ear_dist > 0 else 40
                            cv2.ellipse(mask_canvas, (int(nose['x']*width), int(nose['y']*height)), (hr, int(hr*1.2)), 0, 0, 360, 255, -1, lineType=cv2.LINE_AA)
                        
                        # Face Landmark Rig
                        if include_dense_face:
                            for pt in face: self.draw_point(rig_canvas, pt, (255,255,255), 1, width, height)
                        else:
                            for idxs in FACE_INDICES_SPARSE.values():
                                for idx in idxs:
                                    if idx < len(face): self.draw_point(rig_canvas, face[idx], (255,255,255), 2, width, height)
                        if len(face) > 473:
                            self.draw_point(rig_canvas, face[468], (0, 255, 255), 2, width, height)
                            self.draw_point(rig_canvas, face[473], (0, 255, 255), 2, width, height)

                    # --- Hands Drawing ---
                    if "hands" in frame_2d and frame_2d["hands"]:
                        pose_ref = frame_2d.get("pose", [])
                        r_wrist = pose_ref[16] if len(pose_ref) > 16 else None
                        l_wrist = pose_ref[15] if len(pose_ref) > 15 else None

                        for hand in frame_2d["hands"]:
                            if not hand: continue
                            root = hand[0]
                            
                            # Attempt Snap only if body wrists exist
                            if r_wrist and l_wrist:
                                dist_r = math.hypot(root['x'] - r_wrist['x'], root['y'] - r_wrist['y'])
                                dist_l = math.hypot(root['x'] - l_wrist['x'], root['y'] - l_wrist['y'])
                                if dist_r < dist_l:
                                    target_wrist = r_wrist
                                else:
                                    target_wrist = l_wrist
                                
                                ox, oy = target_wrist['x'] - root['x'], target_wrist['y'] - root['y']
                                for pt in hand:
                                    pt['x'] += ox; pt['y'] += oy

                            # Draw Rig Lines (Keep Gradient for bones)
                            for j, (s_idx, e_idx) in enumerate(HAND_CONNECTIONS):
                                f_idx, j_idx = j // 4, (j % 4) + 1 
                                self.draw_line(rig_canvas, hand[s_idx], hand[e_idx], get_hand_bgr(f_idx, j_idx), 2, width, height)
                            
                            # Draw Rig Points (Force to OpenPose Blue for joints)
                            # BGR for (0, 0, 255) is (255, 0, 0)
                            hand_joint_color_bgr = (255, 0, 0)
                            for j, p in enumerate(hand):
                                self.draw_point(rig_canvas, p, hand_joint_color_bgr, 4, width, height)
                            
                            # Draw Mask
                            self.fill_hand_hull(mask_canvas, hand, color=255)

                    rig_canvas = cv2.cvtColor(rig_canvas, cv2.COLOR_BGR2RGB)
                    rig_frames.append(rig_canvas.astype(np.float32) / 255.0)
                    mask_frames.append(mask_canvas.astype(np.float32) / 255.0)
                    final_pose_data.append(copy.deepcopy(base_frame))

            except Exception as e: 
                print(f"[Yedp] Error processing frame data: {e}")

        if not rig_frames and frames:
            for f in frames:
                rig_frames.append(np.zeros_like(f))
                mask_frames.append(np.zeros(f.shape[:2], dtype=np.float32))

        return (torch.from_numpy(np.array(frames)), torch.from_numpy(np.array(rig_frames)), torch.from_numpy(np.array(mask_frames)), final_pose_data)

# --- CLASS MAPPINGS ---
class YedpWebcamRecorder(YedpMocapBase): pass
class YedpWebcamSnapshot(YedpMocapBase):
    @classmethod
    def INPUT_TYPES(s): 
        types = YedpMocapBase.INPUT_TYPES()
        types["required"]["video_filename"] = ("STRING", {"default": "snapshot.png"})
        return types

class YedpVideoMoCap(YedpMocapBase): pass
class YedpImageMoCap(YedpMocapBase):
    @classmethod
    def INPUT_TYPES(s): 
        types = YedpMocapBase.INPUT_TYPES()
        types["required"]["video_filename"] = ("STRING", {"default": "image.png"})
        return types

class Yedp3DViewer:
    @classmethod
    def INPUT_TYPES(s): return {"required": {"pose_json": ("POSE_DATA",),}}
    RETURN_TYPES = ("POSE_DATA",); RETURN_NAMES = ("pose_json",); FUNCTION = "view_3d"; CATEGORY = "Yedp/MoCap"; OUTPUT_NODE = True
    def view_3d(self, pose_json): return { "ui": { "yedp_3d_data": pose_json }, "result": (pose_json,) }

NODE_CLASS_MAPPINGS = { 
    "YedpWebcamRecorder": YedpWebcamRecorder, "YedpWebcamSnapshot": YedpWebcamSnapshot, 
    "YedpVideoMoCap": YedpVideoMoCap, "YedpImageMoCap": YedpImageMoCap, "Yedp3DViewer": Yedp3DViewer
}

NODE_DISPLAY_NAME_MAPPINGS = { 
    "YedpWebcamRecorder": "Yedp Webcam Recorder (Video)", "YedpWebcamSnapshot": "Yedp Webcam Snapshot (Image)", 
    "YedpVideoMoCap": "Yedp Video MoCap (File)", "YedpImageMoCap": "Yedp Image MoCap (File)", "Yedp3DViewer": "Yedp 3D Viewer"
}
