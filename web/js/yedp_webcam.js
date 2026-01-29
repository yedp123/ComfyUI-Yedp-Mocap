import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// --- GLOBAL CONFIG ---
const BASE_URL = new URL(".", import.meta.url).href;
console.log("[Yedp] Base URL detected:", BASE_URL);

const MEDIAPIPE_JS_URL = new URL("./tasks_vision.js", BASE_URL).href;
const MEDIAPIPE_WASM_DIR = BASE_URL; 

// --- THEME ---
const THEME = {
    primary: "#6a0dad",     
    primaryHover: "#8a2be2",
    stop: "#8b0000",        
    stopHover: "#ff0000",
    text: "#ffffff",
    bg: "rgba(30, 30, 40, 0.8)",
    folderBtn: "#222"
};

// --- FILTERS ---
class OneEuroFilter {
    constructor(minCutoff=1.0, beta=0.0, dCutoff=1.0) {
        this.minCutoff = minCutoff; this.beta = beta; this.dCutoff = dCutoff;
        this.xPrev = this.dxPrev = this.tPrev = null;
    }
    filter(x, t) {
        if(this.tPrev===null){this.xPrev=x;this.dxPrev=0;this.tPrev=t;return x;}
        const tE=t-this.tPrev; if(tE<=0)return this.xPrev;
        const aD=this.smoothingFactor(tE,this.dCutoff), dx=(x-this.xPrev)/tE, dxHat=this.exponentialSmoothing(aD,dx,this.dxPrev);
        const cutoff=this.minCutoff+this.beta*Math.abs(dxHat), a=this.smoothingFactor(tE,cutoff), xHat=this.exponentialSmoothing(a,x,this.xPrev);
        this.xPrev=xHat; this.dxPrev=dxHat; this.tPrev=t; return xHat;
    }
    smoothingFactor(tE, c){const r=2*Math.PI*c*tE;return r/(r+1);}
    exponentialSmoothing(a,x,p){return a*x+(1-a)*p;}
}

class LandmarkSmoother {
    constructor() { this.filters = []; }
    reset() { this.filters = []; }
    smooth(landmarks, timeSec, minCutoff, beta) {
        if(!landmarks) return null;
        if(this.filters.length!==landmarks.length) this.filters=landmarks.map(()=>({x:new OneEuroFilter(minCutoff,beta),y:new OneEuroFilter(minCutoff,beta),z:new OneEuroFilter(minCutoff,beta)}));
        return landmarks.map((p,i)=>({
            x:this.filters[i].x.filter(p.x,timeSec), y:this.filters[i].y.filter(p.y,timeSec), z:this.filters[i].z.filter(p.z,timeSec), visibility:p.visibility
        }));
    }
}

const TASKS = {
    "face": { file: "face_landmarker.task", class: "FaceLandmarker", options: { numFaces:1, minFaceDetectionConfidence:0.5, minFacePresenceConfidence:0.5, minTrackingConfidence:0.5, outputFaceBlendshapes:true } },
    "pose": { file: "pose_landmarker_full.task", class: "PoseLandmarker", options: { numPoses:1, minPoseDetectionConfidence:0.65, minPosePresenceConfidence:0.65, minTrackingConfidence:0.65 } },
    "hand": { file: "hand_landmarker.task", class: "HandLandmarker", options: { numHands:2, minHandDetectionConfidence:0.6, minHandPresenceConfidence:0.6, minTrackingConfidence:0.6 } }
};
const MODE_PRESETS = { "Face + Hands": ["face", "hand"], "Face Only": ["face"], "Face + Body": ["face", "pose"], "Body Only": ["pose"], "Full Holistic (Heavy)": ["face", "pose", "hand"] };

// --- UI HELPERS ---
function createFolderFooter() {
    const footer = document.createElement("div");
    Object.assign(footer.style, { width:"100%", display:"flex", justifyContent:"center", marginTop:"10px", borderTop:"1px solid #444", paddingTop:"5px" });

    const helpText = document.createElement("div");
    helpText.innerText = "⚠️ Remember to clean input/temp folders occasionally.";
    Object.assign(helpText.style, { fontSize:"11px", color:"#ff4444", textAlign:"center", fontWeight:"bold" });

    footer.append(helpText);
    return footer;
}

// =========================================================
// NODE 1: WEBCAM RECORDER
// =========================================================
app.registerExtension({
    name: "Yedp.WebcamRecorder",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "YedpWebcamRecorder") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // --- UI Construction ---
                const container = document.createElement("div");
                Object.assign(container.style, { width:"100%", display:"flex", flexDirection:"column", alignItems:"center", gap:"5px", paddingBottom:"10px" });

                const videoWrapper = document.createElement("div");
                Object.assign(videoWrapper.style, { position:"relative", width:"100%", backgroundColor:"#000", borderRadius:"8px", overflow:"hidden", display:"flex", justifyContent:"center", alignItems:"center", minHeight:"240px", aspectRatio: "16/9" });
                
                const videoEl = document.createElement("video");
                Object.assign(videoEl.style, { width:"100%", height:"100%", objectFit:"contain", transform:"scaleX(-1)", display:"block" });
                videoEl.setAttribute("playsinline", ""); videoEl.muted = true; videoEl.autoplay = true;

                const canvasEl = document.createElement("canvas");
                Object.assign(canvasEl.style, { position:"absolute", top:"0", left:"0", width:"100%", height:"100%", objectFit:"contain", transform:"scaleX(-1)", pointerEvents:"none", zIndex:"5" });

                const overlayEl = document.createElement("div");
                Object.assign(overlayEl.style, { position:"absolute", top:"50%", left:"50%", transform:"translate(-50%, -50%)", fontSize:"64px", fontWeight:"bold", color:"white", textShadow:"0px 0px 20px black", pointerEvents:"none", display:"none", zIndex:"50" });

                const debugText = document.createElement("div");
                Object.assign(debugText.style, { position:"absolute", bottom:"5px", left:"5px", fontSize:"10px", color:"#00FF00", backgroundColor:"rgba(0,0,0,0.6)", padding:"2px 6px", borderRadius:"4px", zIndex:"15", pointerEvents:"none", fontFamily:"monospace" });
                debugText.innerText = "Recorder Ready";

                videoWrapper.append(videoEl, canvasEl, overlayEl, debugText);

                // Settings Row
                const settingsRow = document.createElement("div");
                Object.assign(settingsRow.style, { display:"flex", gap:"5px", width:"100%", flexWrap:"wrap" });
                
                const createSelect = (opts, def, title) => {
                    const s = document.createElement("select");
                    Object.assign(s.style, { flex:"1", padding:"4px", borderRadius:"4px", fontSize:"11px", backgroundColor:"var(--comfy-input-bg)", color:"var(--input-text)", border:"1px solid var(--border-color)", minWidth:"80px" });
                    s.title = title;
                    opts.forEach(o => { const op = document.createElement("option"); op.value = o.value; op.innerText = o.label; if(o.value === def) op.selected = true; s.append(op); });
                    return s;
                };

                const modeSelect = createSelect(Object.keys(MODE_PRESETS).map(k => ({value:k, label:k})), "Face + Hands", "Mode");
                const resSelect = createSelect([{value:"vga", label:"Low Res"}, {value:"hd", label:"High Res"}], "vga", "Res");
                const backendSelect = createSelect([{value:"CPU",label:"CPU"},{value:"GPU",label:"GPU"}], "GPU", "Backend");
                const settingsBtn = document.createElement("button"); settingsBtn.innerText = "⚙️";
                Object.assign(settingsBtn.style, { width:"30px", cursor:"pointer", backgroundColor:"var(--comfy-input-bg)", border:"1px solid var(--border-color)", borderRadius:"4px", color:"var(--input-text)" });

                settingsRow.append(modeSelect, resSelect, backendSelect, settingsBtn);

                // Sliders Panel
                const sliderPanel = document.createElement("div");
                Object.assign(sliderPanel.style, { width:"100%", display:"flex", flexDirection:"column", gap:"5px", padding:"5px", backgroundColor:"rgba(0,0,0,0.2)", borderRadius:"4px", marginTop:"5px" });
                
                const createSlider = (label, min, max, step, val, callback) => {
                    const w = document.createElement("div"); w.style.display="flex"; w.style.alignItems="center"; w.style.fontSize="10px"; w.style.color="var(--input-text)";
                    const l = document.createElement("span"); l.innerText=label; l.style.width="60px";
                    const i = document.createElement("input"); i.type="range"; i.min=min; i.max=max; i.step=step; i.value=val; i.style.flex="1";
                    const v = document.createElement("span"); v.innerText=val; v.style.width="25px"; v.style.textAlign="right";
                    i.oninput = (e) => { v.innerText=e.target.value; callback(parseFloat(e.target.value)); };
                    w.append(l, i, v); return w;
                };
                
                let minCutoff = 0.01; let beta = 20.0;
                sliderPanel.append(createSlider("Jitter", 0.01, 5.0, 0.01, 0.01, v=>minCutoff=v), createSlider("Speed", 0.0, 50.0, 0.1, 20.0, v=>beta=v));
                settingsBtn.onclick = () => sliderPanel.style.display = sliderPanel.style.display === "none" ? "flex" : "none";

                // Action Buttons
                const btnRow = document.createElement("div");
                btnRow.style.display="flex"; btnRow.style.gap="5px"; btnRow.style.width="100%"; btnRow.style.marginTop="5px";
                
                const startBtn = document.createElement("button"); startBtn.innerText = "📷 Start Camera";
                const stopBtn = document.createElement("button"); stopBtn.innerText = "⏹ Stop"; stopBtn.style.display = "none";
                const recBtn = document.createElement("button"); recBtn.innerText = "🔴 Rec"; recBtn.disabled = true;
                
                const styleBtn = (b, primary) => {
                    Object.assign(b.style, { flex:"1", padding:"8px", cursor:"pointer", backgroundColor: primary?THEME.primary:THEME.bg, color:THEME.text, border:"none", borderRadius:"4px", fontWeight:"bold" });
                    b.onmouseenter=()=>b.style.backgroundColor=primary?THEME.primaryHover:"#444"; b.onmouseleave=()=>b.style.backgroundColor=primary?THEME.primary:THEME.bg;
                };
                styleBtn(startBtn, true); styleBtn(stopBtn, false); styleBtn(recBtn, false);

                btnRow.append(startBtn, stopBtn, recBtn);

                // Add Footer
                const footer = createFolderFooter();
                container.append(videoWrapper, settingsRow, sliderPanel, btnRow, footer);

                // Add to Node
                try { let w = this.addDOMWidget("ui", "custom", container); if(w) w.computeSize = w => [w, 440]; } catch(e) { console.error(e); }
                setTimeout(() => { const w=this.widgets?.find(w=>w.name==="video_filename"); if(w?.inputEl) w.inputEl.style.display="none"; }, 100);

                // --- LOGIC ---
                let stream, mediaRecorder, videoChunks=[], jsonChunks=[], isRecording=false, isLooping=false;
                let activeLandmarkers={}, drawingUtils, visionLib, lastTime=-1, frameCount=0;
                const smoothers = { pose: new LandmarkSmoother(), face: new LandmarkSmoother(), hands: [new LandmarkSmoother(), new LandmarkSmoother()] };

                const loadModels = async () => {
                    debugText.innerText = "Loading AI...";
                    try {
                        if(!visionLib) visionLib = await import(MEDIAPIPE_JS_URL);
                        const { FilesetResolver, DrawingUtils } = visionLib;
                        const visionTask = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_DIR);
                        for(let k in activeLandmarkers) activeLandmarkers[k].close();
                        activeLandmarkers={}; smoothers.pose.reset(); smoothers.face.reset();
                        const preset = MODE_PRESETS[modeSelect.value];
                        for(let task of preset) {
                            const conf=TASKS[task]; const url=new URL("./"+conf.file, BASE_URL).href;
                            activeLandmarkers[task] = await visionLib[conf.class].createFromOptions(visionTask, { baseOptions:{modelAssetPath:url, delegate:backendSelect.value}, runningMode:"VIDEO", ...conf.options });
                        }
                        drawingUtils = new DrawingUtils(canvasEl.getContext("2d"));
                        debugText.innerText = "AI Ready"; debugText.style.color = "#00FF00";
                    } catch(e) { console.error(e); debugText.innerText = "Load Error"; debugText.style.color="red"; }
                };

                const render = () => {
                    if(isLooping) requestAnimationFrame(render);
                    if(!videoEl.videoWidth || videoEl.paused) return;
                    if(canvasEl.width!==videoEl.videoWidth){ canvasEl.width=videoEl.videoWidth; canvasEl.height=videoEl.videoHeight; }
                    
                    if(videoEl.currentTime !== lastTime) {
                        lastTime = videoEl.currentTime; const timeMs = performance.now(); const timeSec = timeMs/1000;
                        const ctx = canvasEl.getContext("2d"); ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
                        let frameData = { time: timeMs, frame: ++frameCount };
                        
                        for(let k in activeLandmarkers) {
                            try {
                                const res = activeLandmarkers[k].detectForVideo(videoEl, timeMs);
                                if(k==="face" && res.faceLandmarks.length) {
                                    const smooth = smoothers.face.smooth(res.faceLandmarks[0], timeSec, minCutoff, beta);
                                    if(isRecording) { frameData.face=smooth; if(res.faceBlendshapes) frameData.blendshapes=res.faceBlendshapes[0]; }
                                    drawingUtils.drawConnectors(smooth, visionLib.FaceLandmarker.FACE_LANDMARKS_TESSELATION, {color:"#C0C0C030", lineWidth:1});
                                    drawingUtils.drawConnectors(smooth, visionLib.FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, {color:"#FF3030"});
                                    drawingUtils.drawConnectors(smooth, visionLib.FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, {color:"#30FF30"});
                                    drawingUtils.drawConnectors(smooth, visionLib.FaceLandmarker.FACE_LANDMARKS_LIPS, {color:"#E0E0E0"});
                                }
                                if(k==="pose" && res.landmarks.length) {
                                    const smooth = smoothers.pose.smooth(res.landmarks[0], timeSec, minCutoff, beta);
                                    if(isRecording) { frameData.pose=smooth; frameData.poseWorld=res.worldLandmarks[0]; }
                                    drawingUtils.drawConnectors(smooth, visionLib.PoseLandmarker.POSE_CONNECTIONS, {color:"#00FF00", lineWidth:4});
                                }
                                if(k==="hand" && res.landmarks.length) {
                                    let hData = [];
                                    res.landmarks.forEach((l, i) => { if(smoothers.hands[i]){ const s = smoothers.hands[i].smooth(l, timeSec, minCutoff, beta); hData.push(s); drawingUtils.drawConnectors(s, visionLib.HandLandmarker.HAND_CONNECTIONS, {color:"#00FFFF"}); }});
                                    if(isRecording) frameData.hands=hData;
                                }
                            } catch(e){}
                        }
                        if(isRecording) jsonChunks.push(frameData);
                    }
                };

                const upload = async (blob, json) => {
                    debugText.innerText = "Saving..."; 
                    const name = `yedp_cam_${Date.now()}`; 
                    
                    const fd = new FormData(); fd.append("image", blob, name+".webm"); fd.append("overwrite", "true");
                    const fdJ = new FormData(); fdJ.append("image", new Blob([JSON.stringify(json)],{type:'application/json'}), name+".json"); fdJ.append("overwrite", "true");
                    await api.fetchApi("/upload/image", {method:"POST", body:fd});
                    const w=this.widgets?.find(w=>w.name==="video_filename"); if(w) w.value=name+".webm";
                    await api.fetchApi("/upload/image", {method:"POST", body:fdJ});
                    debugText.innerText = "Saved!";
                };

                recBtn.onclick = () => {
                    if(!isRecording) {
                        let c=3; recBtn.disabled=true; stopBtn.disabled=true;
                        const i=setInterval(()=>{
                            overlayEl.innerText=c; overlayEl.style.display="block"; c--;
                            if(c<0) {
                                clearInterval(i); overlayEl.style.display="none";
                                jsonChunks=[]; videoChunks=[];
                                mediaRecorder=new MediaRecorder(stream, {mimeType:'video/webm'});
                                mediaRecorder.ondataavailable=e=>{if(e.data.size>0)videoChunks.push(e.data)};
                                mediaRecorder.onstop=()=>upload(new Blob(videoChunks,{type:'video/webm'}), jsonChunks);
                                mediaRecorder.start(); isRecording=true; 
                                recBtn.innerText="⬛ Stop"; recBtn.disabled=false; recBtn.style.backgroundColor=THEME.stop;
                            }
                        }, 1000);
                    } else {
                        mediaRecorder.stop(); isRecording=false; recBtn.innerText="🔴 Rec"; recBtn.style.backgroundColor=""; stopBtn.disabled=false;
                    }
                };

                startBtn.onclick = async () => {
                    await loadModels();
                    try {
                        stream = await navigator.mediaDevices.getUserMedia({video: resSelect.value==="hd"?{width:1280,height:720}:{width:640,height:480}});
                        videoEl.srcObject = stream;
                        videoEl.onloadedmetadata=()=>{videoEl.play();isLooping=true;render();};
                        recBtn.disabled=false; startBtn.style.display="none"; stopBtn.style.display="block";
                    } catch(e){alert(e);}
                };
                stopBtn.onclick = () => { isLooping=false; stream.getTracks().forEach(t=>t.stop()); videoEl.srcObject=null; startBtn.style.display="block"; stopBtn.style.display="none"; recBtn.disabled=true; };
                
                // --- Hot-Swap Mode Change ---
                modeSelect.onchange = loadModels; 
                backendSelect.onchange = loadModels;
                
                this.onRemoved = () => { if(stream)stream.getTracks().forEach(t=>t.stop()); isLooping=false; for(let k in activeLandmarkers)activeLandmarkers[k].close(); };
                return r;
            };
        }
    }
});

// =========================================================
// NODE 2: WEBCAM SNAPSHOT (Fixed Hot-Swap + Footer)
// =========================================================
app.registerExtension({
    name: "Yedp.WebcamSnapshot",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "YedpWebcamSnapshot") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                const container = document.createElement("div");
                Object.assign(container.style, { width:"100%", display:"flex", flexDirection:"column", alignItems:"center", gap:"5px", paddingBottom:"10px" });
                
                const videoWrapper = document.createElement("div");
                Object.assign(videoWrapper.style, { position:"relative", width:"100%", backgroundColor:"#000", borderRadius:"8px", overflow:"hidden", minHeight:"240px", aspectRatio:"16/9" });
                const videoEl = document.createElement("video");
                Object.assign(videoEl.style, { width:"100%", height:"100%", objectFit:"contain", transform:"scaleX(-1)", display:"block" });
                videoEl.setAttribute("playsinline", ""); videoEl.muted=true; videoEl.autoplay=true;
                const canvasEl = document.createElement("canvas");
                Object.assign(canvasEl.style, { position:"absolute", top:"0", left:"0", width:"100%", height:"100%", objectFit:"contain", transform:"scaleX(-1)", pointerEvents:"none", zIndex:"5" });
                const overlayEl = document.createElement("div");
                Object.assign(overlayEl.style, { position:"absolute", top:"50%", left:"50%", transform:"translate(-50%, -50%)", fontSize:"64px", fontWeight:"bold", color:"white", textShadow:"0px 0px 20px black", pointerEvents:"none", display:"none", zIndex:"50" });
                
                const debugText = document.createElement("div");
                Object.assign(debugText.style, { position:"absolute", bottom:"5px", left:"5px", fontSize:"10px", color:"#00FF00", backgroundColor:"rgba(0,0,0,0.6)", padding:"2px 6px", borderRadius:"4px", zIndex:"15", pointerEvents:"none", fontFamily:"monospace" });
                debugText.innerText = "Snapshot Ready";
                
                videoWrapper.append(videoEl, canvasEl, overlayEl, debugText);

                const settingsRow = document.createElement("div");
                Object.assign(settingsRow.style, { display:"flex", gap:"5px", width:"100%", flexWrap:"wrap" });
                const createSelect = (opts, def, title) => { const s=document.createElement("select"); Object.assign(s.style, {flex:"1", padding:"4px", fontSize:"11px", backgroundColor:"var(--comfy-input-bg)", color:"var(--input-text)", border:"1px solid var(--border-color)"}); s.title=title; opts.forEach(o=>{const op=document.createElement("option"); op.value=o.value; op.innerText=o.label; if(o.value===def)op.selected=true; s.append(op);}); return s; };
                const modeSelect = createSelect(Object.keys(MODE_PRESETS).map(k=>({value:k, label:k})), "Face + Hands");
                const backendSelect = createSelect([{value:"CPU",label:"CPU"},{value:"GPU",label:"GPU"}], "GPU");
                settingsRow.append(modeSelect, backendSelect);

                const btnRow = document.createElement("div");
                btnRow.style.display="flex"; btnRow.style.gap="5px"; btnRow.style.width="100%"; btnRow.style.marginTop="5px";
                const startBtn = document.createElement("button"); startBtn.innerText = "📷 Start";
                const stopBtn = document.createElement("button"); stopBtn.innerText = "⏹ Stop"; stopBtn.style.display="none";
                const snapBtn = document.createElement("button"); snapBtn.innerText = "📸 Snap"; snapBtn.disabled = true;
                
                const styleBtn = (b, primary) => {
                    Object.assign(b.style, { flex:"1", padding:"8px", cursor:"pointer", backgroundColor: primary?THEME.primary:THEME.bg, color:THEME.text, border:"none", borderRadius:"4px", fontWeight:"bold" });
                    b.onmouseenter=()=>b.style.backgroundColor=primary?THEME.primaryHover:"#444"; b.onmouseleave=()=>b.style.backgroundColor=primary?THEME.primary:THEME.bg;
                };
                styleBtn(startBtn, true); styleBtn(stopBtn, false); styleBtn(snapBtn, false);
                btnRow.append(startBtn, stopBtn, snapBtn);

                // Add Footer
                const footer = createFolderFooter();
                container.append(videoWrapper, settingsRow, btnRow, footer);

                try { let w = this.addDOMWidget("ui", "custom", container); if(w) w.computeSize=w=>[w, 420]; } catch(e){}
                setTimeout(() => { const w=this.widgets?.find(w=>w.name==="video_filename"); if(w?.inputEl) w.inputEl.style.display="none"; }, 100);

                let stream, activeLandmarkers={}, visionLib, isLooping=false, lastTime=-1, drawingUtils;
                
                // --- FIXED LOAD MODELS (Updates State In-Place for Hot-Swap) ---
                const loadModels = async () => {
                    debugText.innerText = "Loading AI...";
                    try {
                        if(!visionLib) visionLib = await import(MEDIAPIPE_JS_URL);
                        const { FilesetResolver, DrawingUtils } = visionLib;
                        const visionTask = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_DIR);
                        for(let k in activeLandmarkers) activeLandmarkers[k].close();
                        activeLandmarkers={};
                        for(let task of MODE_PRESETS[modeSelect.value]) {
                            const conf=TASKS[task]; const url=new URL("./"+conf.file, BASE_URL).href;
                            activeLandmarkers[task] = await visionLib[conf.class].createFromOptions(visionTask, { baseOptions:{modelAssetPath:url, delegate:backendSelect.value}, runningMode:"VIDEO", ...conf.options });
                        }
                        drawingUtils = new DrawingUtils(canvasEl.getContext("2d"));
                        debugText.innerText = "AI Ready";
                    } catch(e) { console.error(e); debugText.innerText = "Load Error"; }
                };

                const render = () => {
                    if(isLooping) requestAnimationFrame(render);
                    if(!videoEl.videoWidth || videoEl.paused) return;
                    if(canvasEl.width!==videoEl.videoWidth){ canvasEl.width=videoEl.videoWidth; canvasEl.height=videoEl.videoHeight; }
                    if(videoEl.currentTime!==lastTime) {
                        lastTime = videoEl.currentTime; const timeMs=performance.now();
                        const ctx=canvasEl.getContext("2d"); ctx.clearRect(0,0,canvasEl.width,canvasEl.height);
                        for(let k in activeLandmarkers) {
                            try {
                                const res = activeLandmarkers[k].detectForVideo(videoEl, timeMs);
                                if(drawingUtils) {
                                    if(k==="face" && res.faceLandmarks.length) drawingUtils.drawConnectors(res.faceLandmarks[0], visionLib.FaceLandmarker.FACE_LANDMARKS_TESSELATION, {color:"#C0C0C030", lineWidth:1});
                                    if(k==="pose" && res.landmarks.length) drawingUtils.drawConnectors(res.landmarks[0], visionLib.PoseLandmarker.POSE_CONNECTIONS, {color:"#00FF00", lineWidth:4});
                                    if(k==="hand" && res.landmarks.length) res.landmarks.forEach(l => drawingUtils.drawConnectors(l, visionLib.HandLandmarker.HAND_CONNECTIONS, {color:"#00FFFF"}));
                                }
                            } catch(e){}
                        }
                    }
                };

                startBtn.onclick = async () => {
                    await loadModels();
                    try {
                        stream = await navigator.mediaDevices.getUserMedia({video:{width:1280,height:720}});
                        videoEl.srcObject = stream;
                        videoEl.onloadedmetadata = () => { videoEl.play(); isLooping=true; render(); };
                        snapBtn.disabled=false; startBtn.style.display="none"; stopBtn.style.display="block";
                    } catch(e){ alert(e); }
                };

                stopBtn.onclick = () => { 
                    isLooping=false; if(stream)stream.getTracks().forEach(t=>t.stop()); videoEl.srcObject=null; 
                    startBtn.style.display="block"; stopBtn.style.display="none"; snapBtn.disabled=true;
                };

                snapBtn.onclick = async () => {
                    snapBtn.disabled = true; stopBtn.disabled = true;
                    let c = 3;
                    const i = setInterval(() => {
                        overlayEl.innerText = c; overlayEl.style.display = "block"; c--;
                        if(c < 0) {
                            clearInterval(i); overlayEl.style.display = "none";
                            // Capture
                            const name = `yedp_snap_${Date.now()}`;
                            debugText.innerText = "Capturing...";
                            let frameData = { time: 0, frame: 1 };
                            const timeMs = performance.now();
                            for(let k in activeLandmarkers) {
                                try {
                                    const res = activeLandmarkers[k].detectForVideo(videoEl, timeMs);
                                    if(k==="face" && res.faceLandmarks.length) { frameData.face=res.faceLandmarks[0]; if(res.faceBlendshapes)frameData.blendshapes=res.faceBlendshapes[0]; }
                                    if(k==="pose" && res.landmarks.length) { frameData.pose=res.landmarks[0]; frameData.poseWorld=res.worldLandmarks[0]; }
                                    if(k==="hand" && res.landmarks.length) frameData.hands=res.landmarks;
                                } catch(e){}
                            }
                            const tmp = document.createElement("canvas"); tmp.width = videoEl.videoWidth; tmp.height = videoEl.videoHeight;
                            const ctx = tmp.getContext("2d"); ctx.translate(tmp.width, 0); ctx.scale(-1, 1); ctx.drawImage(videoEl, 0, 0);
                            
                            tmp.toBlob(async (blob) => {
                                const fd = new FormData(); fd.append("image", blob, name+".png"); fd.append("overwrite", "true");
                                const fdJ = new FormData(); fdJ.append("image", new Blob([JSON.stringify(frameData)], {type:'application/json'}), name+".json"); fdJ.append("overwrite", "true");
                                await api.fetchApi("/upload/image", {method:"POST", body:fd});
                                const w=this.widgets?.find(w=>w.name==="video_filename"); if(w) w.value=name+".png";
                                await api.fetchApi("/upload/image", {method:"POST", body:fdJ});
                                debugText.innerText = "Snapshot Saved!";
                                snapBtn.disabled = false; stopBtn.disabled = false;
                            }, "image/png");
                        }
                    }, 1000);
                };
                
                // --- HOT SWAP ENABLED ---
                modeSelect.onchange = loadModels;
                backendSelect.onchange = loadModels;

                this.onRemoved = () => { if(stream)stream.getTracks().forEach(t=>t.stop()); isLooping=false; for(let k in activeLandmarkers)activeLandmarkers[k].close(); };
                return r;
            };
        }
    }
});

// =========================================================
// NODE 3: VIDEO FILE MOCAP
// =========================================================
app.registerExtension({
    name: "Yedp.VideoMoCap",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "YedpVideoMoCap") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                const container = document.createElement("div");
                Object.assign(container.style, { width:"100%", display:"flex", flexDirection:"column", alignItems:"center", gap:"5px", paddingBottom:"10px" });
                
                const videoWrapper = document.createElement("div");
                Object.assign(videoWrapper.style, { position:"relative", width:"100%", backgroundColor:"#000", borderRadius:"8px", overflow:"hidden", minHeight:"240px", aspectRatio:"16/9" });
                const videoEl = document.createElement("video");
                Object.assign(videoEl.style, { width:"100%", height:"100%", objectFit:"contain", display:"block" });
                videoEl.controls = true;
                const canvasEl = document.createElement("canvas");
                Object.assign(canvasEl.style, { position:"absolute", top:"0", left:"0", width:"100%", height:"100%", objectFit:"contain", pointerEvents:"none", zIndex:"5" });
                const debugText = document.createElement("div");
                Object.assign(debugText.style, { position:"absolute", bottom:"30px", left:"5px", fontSize:"10px", color:"#00FF00", backgroundColor:"rgba(0,0,0,0.6)", padding:"2px 6px", borderRadius:"4px", zIndex:"15", pointerEvents:"none", fontFamily:"monospace" });
                debugText.innerText = "Load Video File";
                
                videoWrapper.append(videoEl, canvasEl, debugText);

                const settingsRow = document.createElement("div");
                Object.assign(settingsRow.style, { display:"flex", gap:"5px", width:"100%" });
                const createSelect = (opts, def) => { const s=document.createElement("select"); Object.assign(s.style, {flex:"1", padding:"4px", fontSize:"11px", backgroundColor:"var(--comfy-input-bg)", color:"var(--input-text)", border:"1px solid var(--border-color)"}); opts.forEach(o=>{const op=document.createElement("option"); op.value=o.value; op.innerText=o.label; if(o.value===def)op.selected=true; s.append(op);}); return s; };
                const modeSelect = createSelect(Object.keys(MODE_PRESETS).map(k=>({value:k, label:k})), "Full Holistic (Heavy)");
                const backendSelect = createSelect([{value:"CPU",label:"CPU"},{value:"GPU",label:"GPU"}], "GPU");
                settingsRow.append(modeSelect, backendSelect);

                const fileInput = document.createElement("input"); fileInput.type="file"; fileInput.accept="video/*"; fileInput.style.display="none";
                
                const btnRow = document.createElement("div");
                btnRow.style.display="flex"; btnRow.style.gap="5px"; btnRow.style.width="100%"; btnRow.style.marginTop="5px";
                const loadBtn = document.createElement("button"); loadBtn.innerText = "📂 Load";
                const resetBtn = document.createElement("button"); resetBtn.innerText = "🔄 Reset";
                const processBtn = document.createElement("button"); processBtn.innerText = "▶ Process"; processBtn.disabled=true;
                
                const styleBtn = (b, primary) => {
                    Object.assign(b.style, { flex:"1", padding:"8px", cursor:"pointer", backgroundColor: primary?THEME.primary:THEME.bg, color:THEME.text, border:"none", borderRadius:"4px", fontWeight:"bold" });
                    b.onmouseenter=()=>b.style.backgroundColor=primary?THEME.primaryHover:"#444"; b.onmouseleave=()=>b.style.backgroundColor=primary?THEME.primary:THEME.bg;
                };
                styleBtn(loadBtn, false); styleBtn(resetBtn, false); styleBtn(processBtn, true);
                btnRow.append(loadBtn, resetBtn, processBtn);

                // Add Footer
                const footer = createFolderFooter();
                container.append(videoWrapper, settingsRow, fileInput, btnRow, footer);

                try { let w = this.addDOMWidget("ui", "custom", container); if(w) w.computeSize=w=>[w, 400]; } catch(e){}
                setTimeout(() => { const w=this.widgets?.find(w=>w.name==="video_filename"); if(w?.inputEl) w.inputEl.style.display="none"; }, 100);

                let activeLandmarkers={}, visionLib, drawingUtils, jsonChunks=[];
                
                // Helper to load models
                const loadModels = async () => {
                    debugText.innerText = "Loading Models...";
                    try {
                        if(!visionLib) visionLib = await import(MEDIAPIPE_JS_URL);
                        const { FilesetResolver, DrawingUtils } = visionLib;
                        const visionTask = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_DIR);
                        for(let k in activeLandmarkers) activeLandmarkers[k].close();
                        activeLandmarkers={};
                        const preset = MODE_PRESETS[modeSelect.value];
                        for(let task of preset) {
                            const conf=TASKS[task]; const url=new URL("./"+conf.file, BASE_URL).href;
                            activeLandmarkers[task] = await visionLib[conf.class].createFromOptions(visionTask, { baseOptions:{modelAssetPath:url, delegate:backendSelect.value}, runningMode:"VIDEO", ...conf.options });
                        }
                        drawingUtils = new DrawingUtils(canvasEl.getContext("2d"));
                        debugText.innerText = "Models Ready";
                    } catch(e) { console.error(e); debugText.innerText="Load Failed"; }
                };
                
                loadBtn.onclick = () => fileInput.click();
                fileInput.onchange = async (e) => {
                    const file = e.target.files[0]; if(!file) return;
                    await loadModels(); // Ensure models loaded for this file
                    videoEl.src = URL.createObjectURL(file);
                    processBtn.disabled = false;
                    debugText.innerText = "Loaded: " + file.name;
                };

                resetBtn.onclick = () => {
                    videoEl.pause(); videoEl.removeAttribute('src'); videoEl.load();
                    fileInput.value = ""; processBtn.disabled = true;
                    canvasEl.getContext("2d").clearRect(0,0,canvasEl.width,canvasEl.height);
                    debugText.innerText = "Cleared";
                };

                processBtn.onclick = async () => {
                    // CRITICAL FIX: Ensure models match current dropdown selection
                    await loadModels();

                    processBtn.disabled = true; loadBtn.disabled = true;
                    jsonChunks = []; videoEl.pause(); videoEl.currentTime = 0;
                    
                    await new Promise(r => setTimeout(r, 500));
                    
                    const duration = videoEl.duration;
                    const FPS = 30;
                    const step = 1/FPS;
                    let currentTime = 0;
                    
                    canvasEl.width = videoEl.videoWidth || 1280;
                    canvasEl.height = videoEl.videoHeight || 720;
                    const ctx = canvasEl.getContext("2d");

                    const processFrame = async () => {
                        if (currentTime >= duration) {
                            debugText.innerText = "Uploading...";
                            const timestamp = Date.now();
                            const name = `yedp_vid_${timestamp}`; 
                            try {
                                let ext = ".webm"; if (fileInput.files[0]) { const n = fileInput.files[0].name; ext = n.substring(n.lastIndexOf('.')); }
                                const blob = await fetch(videoEl.src).then(r => r.blob());
                                const fd = new FormData(); fd.append("image", blob, name + ext); fd.append("overwrite", "true");
                                const fdJ = new FormData(); fdJ.append("image", new Blob([JSON.stringify(jsonChunks)], {type:'application/json'}), name+".json"); fdJ.append("overwrite", "true");
                                await api.fetchApi("/upload/image", {method:"POST", body:fd});
                                const w=this.widgets?.find(w=>w.name==="video_filename"); if(w) w.value=name + ext;
                                await api.fetchApi("/upload/image", {method:"POST", body:fdJ});
                                debugText.innerText = "Done! Ready to Queue";
                                processBtn.disabled = false; loadBtn.disabled = false;
                            } catch(e) { console.error(e); debugText.innerText = "Upload Failed"; }
                            return;
                        }

                        // Robust Seek
                        await new Promise(r => {
                            let done = false;
                            const onSeek = () => { if(done)return; done=true; videoEl.removeEventListener('seeked', onSeek); r(); };
                            videoEl.addEventListener('seeked', onSeek);
                            videoEl.currentTime = currentTime;
                            setTimeout(() => { if(!done) onSeek(); }, 500);
                        });

                        ctx.clearRect(0,0,canvasEl.width,canvasEl.height);
                        const timeMs = currentTime * 1000;
                        let frameData = { time: timeMs, frame: jsonChunks.length };
                        
                        for(let k in activeLandmarkers) {
                            try {
                                const res = activeLandmarkers[k].detectForVideo(videoEl, timeMs);
                                if(k==="face" && res.faceLandmarks.length) { frameData.face=res.faceLandmarks[0]; drawingUtils.drawConnectors(res.faceLandmarks[0], visionLib.FaceLandmarker.FACE_LANDMARKS_TESSELATION, {color:"#C0C0C030", lineWidth:1}); }
                                if(k==="pose" && res.landmarks.length) { frameData.pose=res.landmarks[0]; drawingUtils.drawConnectors(res.landmarks[0], visionLib.PoseLandmarker.POSE_CONNECTIONS, {color:"#00FF00", lineWidth:2}); }
                                if(k==="hand" && res.landmarks.length) { frameData.hands=res.landmarks; res.landmarks.forEach(l => drawingUtils.drawConnectors(l, visionLib.HandLandmarker.HAND_CONNECTIONS, {color:"#00FFFF"})); }
                            } catch(e){}
                        }
                        jsonChunks.push(frameData);
                        debugText.innerText = `Proc: ${Math.round((currentTime/duration)*100)}%`;
                        currentTime += step;
                        setTimeout(processFrame, 5);
                    };
                    processFrame();
                };
                
                this.onRemoved = () => { for(let k in activeLandmarkers)activeLandmarkers[k].close(); };
                return r;
            };
        }
    }
});

// =========================================================
// NODE 4: IMAGE FILE MOCAP
// =========================================================
app.registerExtension({
    name: "Yedp.ImageMoCap",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "YedpImageMoCap") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                const container = document.createElement("div");
                Object.assign(container.style, { width:"100%", display:"flex", flexDirection:"column", alignItems:"center", gap:"5px", paddingBottom:"10px" });
                
                const imgWrapper = document.createElement("div");
                Object.assign(imgWrapper.style, { position:"relative", width:"100%", backgroundColor:"#000", borderRadius:"8px", overflow:"hidden", minHeight:"240px", aspectRatio:"16/9" });
                const imgEl = document.createElement("img");
                Object.assign(imgEl.style, { width:"100%", height:"100%", objectFit:"contain", display:"block" });
                
                const canvasEl = document.createElement("canvas");
                Object.assign(canvasEl.style, { position:"absolute", top:"0", left:"0", width:"100%", height:"100%", objectFit:"contain", pointerEvents:"none", zIndex:"5" });
                const debugText = document.createElement("div");
                Object.assign(debugText.style, { position:"absolute", bottom:"5px", left:"5px", fontSize:"10px", color:"#00FF00", backgroundColor:"rgba(0,0,0,0.6)", padding:"2px 6px", borderRadius:"4px", zIndex:"15", pointerEvents:"none", fontFamily:"monospace" });
                debugText.innerText = "Load Image File";
                
                imgWrapper.append(imgEl, canvasEl, debugText);

                const settingsRow = document.createElement("div");
                Object.assign(settingsRow.style, { display:"flex", gap:"5px", width:"100%" });
                const createSelect = (opts, def) => { const s=document.createElement("select"); Object.assign(s.style, {flex:"1", padding:"4px", fontSize:"11px", backgroundColor:"var(--comfy-input-bg)", color:"var(--input-text)", border:"1px solid var(--border-color)"}); opts.forEach(o=>{const op=document.createElement("option"); op.value=o.value; op.innerText=o.label; if(o.value===def)op.selected=true; s.append(op);}); return s; };
                const modeSelect = createSelect(Object.keys(MODE_PRESETS).map(k=>({value:k, label:k})), "Full Holistic (Heavy)");
                const backendSelect = createSelect([{value:"CPU",label:"CPU"},{value:"GPU",label:"GPU"}], "GPU");
                settingsRow.append(modeSelect, backendSelect);

                const fileInput = document.createElement("input"); fileInput.type="file"; fileInput.accept="image/*"; fileInput.style.display="none";
                
                const btnRow = document.createElement("div");
                btnRow.style.display="flex"; btnRow.style.gap="5px"; btnRow.style.width="100%"; btnRow.style.marginTop="5px";
                const loadBtn = document.createElement("button"); loadBtn.innerText = "📂 Load Image";
                const processBtn = document.createElement("button"); processBtn.innerText = "▶ Process"; processBtn.disabled=true;
                
                const styleBtn = (b, primary) => {
                    Object.assign(b.style, { flex:"1", padding:"8px", cursor:"pointer", backgroundColor: primary?THEME.primary:THEME.bg, color:THEME.text, border:"none", borderRadius:"4px", fontWeight:"bold" });
                    b.onmouseenter=()=>b.style.backgroundColor=primary?THEME.primaryHover:"#444"; b.onmouseleave=()=>b.style.backgroundColor=primary?THEME.primary:THEME.bg;
                };
                styleBtn(loadBtn, false); styleBtn(processBtn, true);
                btnRow.append(loadBtn, processBtn);

                // Add Footer
                const footer = createFolderFooter();
                container.append(imgWrapper, settingsRow, fileInput, btnRow, footer);

                try { let w = this.addDOMWidget("ui", "custom", container); if(w) { w.element=container; w.computeSize=(w)=>[w, 400]; } } catch(e){}
                setTimeout(() => { const w=this.widgets?.find(w=>w.name==="video_filename"); if(w?.inputEl) w.inputEl.style.display="none"; }, 100);

                let activeLandmarkers={}, visionLib, drawingUtils, jsonChunks=[];
                
                const reloadModels = async () => {
                    debugText.innerText = "Loading Models...";
                    try {
                        if(!visionLib) visionLib = await import(MEDIAPIPE_JS_URL);
                        const { FilesetResolver, DrawingUtils } = visionLib;
                        const visionTask = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_DIR);
                        for(let k in activeLandmarkers) activeLandmarkers[k].close();
                        activeLandmarkers={};
                        const preset = MODE_PRESETS[modeSelect.value];
                        for(let task of preset) {
                            const conf=TASKS[task]; const url=new URL("./"+conf.file, BASE_URL).href;
                            // For images we MUST use IMAGE mode
                            activeLandmarkers[task] = await visionLib[conf.class].createFromOptions(visionTask, { baseOptions:{modelAssetPath:url, delegate:backendSelect.value}, runningMode:"IMAGE", ...conf.options });
                        }
                        drawingUtils = new DrawingUtils(canvasEl.getContext("2d"));
                        if (imgEl.src) { processBtn.disabled = false; debugText.innerText = "Ready to Process"; }
                    } catch(e) { console.error(e); debugText.innerText="Model Load Failed"; }
                };

                // Reload on change
                modeSelect.onchange = reloadModels;
                backendSelect.onchange = reloadModels;

                loadBtn.onclick = () => fileInput.click();
                fileInput.onchange = async (e) => {
                    const file = e.target.files[0]; if(!file) return;
                    imgEl.src = URL.createObjectURL(file);
                    // Load models if not yet loaded or just to be safe
                    await reloadModels();
                    debugText.innerText = "Loaded: " + file.name;
                };

                processBtn.onclick = async () => {
                    debugText.innerText = "Processing...";
                    jsonChunks = [];
                    if(canvasEl.width!==imgEl.naturalWidth){ canvasEl.width=imgEl.naturalWidth; canvasEl.height=imgEl.naturalHeight; }
                    const ctx=canvasEl.getContext("2d"); ctx.clearRect(0,0,canvasEl.width,canvasEl.height);
                    
                    let frameData = { time: 0, frame: 0 };
                    for(let k in activeLandmarkers) {
                        try {
                            const res = activeLandmarkers[k].detect(imgEl); 
                            if(k==="face" && res.faceLandmarks.length) { 
                                frameData.face=res.faceLandmarks[0];
                                drawingUtils.drawConnectors(res.faceLandmarks[0], visionLib.FaceLandmarker.FACE_LANDMARKS_TESSELATION, {color:"#C0C0C030", lineWidth:1});
                            }
                            if(k==="pose" && res.landmarks.length) {
                                frameData.pose=res.landmarks[0];
                                drawingUtils.drawConnectors(res.landmarks[0], visionLib.PoseLandmarker.POSE_CONNECTIONS, {color:"#00FF00", lineWidth:2});
                            }
                            if(k==="hand" && res.landmarks.length) {
                                frameData.hands=res.landmarks;
                                res.landmarks.forEach(l => drawingUtils.drawConnectors(l, visionLib.HandLandmarker.HAND_CONNECTIONS, {color:"#00FFFF"}));
                            }
                        } catch(e){ console.error(e); }
                    }
                    
                    debugText.innerText = "Uploading...";
                    const name = `yedp_img_${Date.now()}`;
                    
                    let ext = ".png";
                    if (fileInput.files[0]) { const n = fileInput.files[0].name; ext = n.substring(n.lastIndexOf('.')); }

                    const blob = await fetch(imgEl.src).then(r => r.blob());
                    const fd = new FormData(); fd.append("image", blob, name + ext); fd.append("overwrite", "true");
                    const fdJ = new FormData(); fdJ.append("image", new Blob([JSON.stringify([frameData])], {type:'application/json'}), name+".json"); fdJ.append("overwrite", "true");
                    
                    await api.fetchApi("/upload/image", {method:"POST", body:fd});
                    const w=this.widgets?.find(w=>w.name==="video_filename"); if(w) w.value=name + ext;
                    await api.fetchApi("/upload/image", {method:"POST", body:fdJ});
                    
                    debugText.innerText = "Done! Ready to Queue";
                };
                
                this.onRemoved = () => { for(let k in activeLandmarkers)activeLandmarkers[k].close(); };
                return r;
            };
        }
    }
});