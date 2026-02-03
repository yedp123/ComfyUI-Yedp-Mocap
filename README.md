**ComfyUI-Yedp-MoCap (Experimental)**

<img width="1077" height="822" alt="Preview" src="https://github.com/user-attachments/assets/fde25ee0-d44b-40c7-8cb8-a0e787dcb58a" />


A Browser-based Motion Capture Studio for ComfyUI.

ComfyUI-Yedp-MoCap moves the heavy lifting of Pose, Hand, and Face detection from your GPU (Python) to your Browser (JavaScript/MediaPipe). This allows for Real-Time feedback before recording and saves your VRAM for the actual generation.



🌟 <ins>**Why use this?**</ins>

- See the skeleton overlay LIVE on your webcam feed. Ensure your hands are in frame before you hit record.

- Zero VRAM Cost: Detection runs on your CPU/Integrated Graphics via the browser. This leaves 100% of your Nvidia VRAM free for Stable Diffusion, ControlNet, and AnimateDiff.

- OpenPose Compatible: The node automatically converts MediaPipe data into standard OpenPose Rainbow format (Correct finger colors, minimalist face features) ready for ControlNet.

- 3D Bridge: Exports raw x,y,z coordinate data to .json.



📦 <ins>**The 5 Nodes**</ins>


1. Yedp Webcam Recorder (The Main Tool)
Function: Records video clips from your webcam with a countdown timer.
Best for: Video-to-Video, AnimateDiff workflows.
Features: Live preview, 1-Euro smoothing filters, adjustable resolution.

2. Yedp Webcam Snapshot
Function: Captures a single still image with pose data.
Best for: Creating reference poses for ControlNet or Image-to-Image.

4. Yedp Video MoCap
Function: Loads an existing video file and processes it frame-by-frame.
Feature: Uses a "Seek-and-Wait" logic to ensure 100% frame accuracy, even on slower computers.

4. Yedp Image MoCap
Function: Load any image to extract the pose/rig.

5. Yedp 3D Pose Viewer
Function: Takes the POSE_DATA (JSON) output from any of the nodes above and renders it in a 3D viewport.
Usage: Verify depth and movement in 3D space before exporting to external software.



🛠️ <ins>**Installation**</ins>

1. Navigate to your ComfyUI/custom_nodes/ folder in your terminal/cmd.

2. Clone this repository:

git clone [https://github.com/yedp123/ComfyUI-Yedp-MoCap](https://github.com/yedp123/ComfyUI-Yedp-MoCap)

3. Restart ComfyUI.


Note: All necessary AI models (.task files) are bundled in the web/js folder. No manual downloads required.



⚙️ <ins>**Usage Tips**</ins>

The Settings:


- Tracking Mode:
  
Face + Hands: Best for desk usage (Streaming/Talking).
Full Holistic: Full body + Face + Hands (Heavy).


- Smoothing Sliders:

Jitter: Increases stability for shaky hands.
Speed/Lag: Controls how fast the skeleton follows you. (Lower = Tighter sync, Higher = Smoother).


- The Outputs

IMAGE: The raw video/image.
RIG_IMAGE: The colored OpenPose map. Plug this directly into ControlNet Apply.
POSE_DATA: The raw JSON. Save this using a "Save Text" node if you want to import it into Blender.
MASK: A volumetric white silhouette. (Note: This is experimental/rough. Use BiRefNet or RemBG if you need a perfect composite matte).



⚠️ <ins>**Known Issues / Limitations**</ins>


Browser Dependency: This node relies on your browser's performance. Chrome/Edge recommended.
Mask Quality: The mask is generated from bone volume (cylinders), so it won't capture loose clothing or hair accurately.
Audio: Audio export is currently disabled to ensure stability.






**CREDITS**

Original Concept by Yedp. Built with MediaPipe (Google) and ComfyUI.
