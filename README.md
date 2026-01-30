ComfyUI-Yedp-MoCap (Experimental)
A client-side MediaPipe implementation for ComfyUI that offers Real-Time Motion Capture feedback.





⚠️ **Experimental Node**

This is an experimental release. It moves the heavy lifting of Pose/Face detection from your GPU (Python) to your Browser (JavaScript).

Pros: Zero VRAM usage during detection, Real-time feedback, JSON export for 3D apps.

Cons: The "Mask" output is rough (volumetric only).


🌟 **Key Features**

Record yourself via Webcam directly inside the node. See the skeleton overlay LIVE before you record to ensure your framing is perfect.
VRAM Saver: Detection runs on your CPU/Browser. This saves your precious VRAM for the actual generation (ControlNet + AnimateDiff).
JSON Bridge: Exports raw x,y,z coordinate data to .json. Useful for 3D Generalists who want to drive rigs with the captured data.

3 Modes: 

- Webcam Recorder: For video-to-video workflows.
- Video Loader: Process existing video files with frame-by-frame precision.
- Snapshot: Capture static poses for image generation.

  
🛠️ **Installation**

Clone this repository into your ComfyUI/custom_nodes/ folder:
cd ComfyUI/custom_nodes/
git clone [https://github.com/yourusername/ComfyUI-Yedp-MoCap](https://github.com/yourusername/ComfyUI-Yedp-MoCap)

Restart ComfyUI.
That's it! The MediaPipe models are included in the web/js folder, so no manual download is required.


📖 **Usage**

1. Yedp Webcam Recorder
Best for: Vid2Vid, AnimateDiff workflows.
How: Click Start, frame yourself using the green/red skeleton overlay. Click Rec. A 3-second timer will start, then recording begins. Click Stop to save.
Outputs: The video and JSON are saved to your input folder automatically.

2. Yedp Video MoCap
Best for: Processing stock footage or pre-recorded clips.
How: Click Load Video. Select a file. Click Process.
Note: The node uses a "Seek-and-Wait" method to ensure perfect synchronization. It may look slow, but it guarantees no frames are skipped.

3. Settings (The ⚙️ Button)
Tracking Mode: Switch between "Face Only," "Face + Hands," or "Full Holistic."
Tip: "Face + Hands" is usually best for desk/webcam setups.
Jitter Fix: Increases smoothing for shaky hands/bodies.
Backend: Switch between CPU and GPU (Browser-side). CPU is often more stable for multi-model tracking.


📦 **Outputs**

- IMAGE: The raw video frames (Webcam feed).
- RIG_IMAGE: A black background with standard OpenPose colored skeletons. Plug this directly into ControlNet Apply.
- MASK: A rough, volumetric matte (White silhouette). Useful for basic composition, but use BiRefNet or RemBG if you need high-quality rotoscoping.
- POSE_DATA: The raw JSON coordinate data.


🐛 **Known Issues / To-Do**

Audio: Audio export is currently disabled to prevent crashes.
Mask Quality: The mask is generated from bone volume, so it won't capture loose clothing or hair accurately.
Browser: Tested primarily on Chrome/Edge. Firefox may behave differently with GPU delegates.


**Credits**

Original concept by Yedp, built with MediaPipe (Google) and ComfyUI.
