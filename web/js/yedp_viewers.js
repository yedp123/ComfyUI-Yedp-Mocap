import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// --- THEME ---
const THEME = {
    bg: "#222",
    accent: "#6a0dad",
    text: "#ddd",
    btn: "#333",
    btnHover: "#444"
};

// --- HELPER: QUEUE ANCESTORS ONLY ---
// Traverses the graph backwards from the target node to create a partial execution graph.
const queueNodeAncestors = async (node) => {
    // 1. Get the full graph serialization
    const p = await app.graphToPrompt(); 
    const output = p.output; // The prompt structure (node_id -> { inputs, class_type })
    const workflow = p.workflow;

    // 2. Traverse backwards to find dependencies
    const requiredIds = new Set();
    // Start with the current node
    const stack = [node.id.toString()]; 

    while(stack.length > 0) {
        const currentId = stack.pop();
        if(requiredIds.has(currentId)) continue;
        requiredIds.add(currentId);

        // Retrieve node data from the full graph
        const nodeData = output[currentId];
        if(!nodeData) continue;

        const inputs = nodeData.inputs;
        if(!inputs) continue;

        // Check all inputs for connections
        for(const key in inputs) {
            const val = inputs[key];
            // In ComfyUI API, connections are arrays: ["parent_id", output_index]
            if(Array.isArray(val) && val.length === 2) {
                const parentId = val[0].toString();
                stack.push(parentId);
            }
        }
    }

    // 3. Construct the partial output graph
    const filteredOutput = {};
    for(const id in output) {
        if(requiredIds.has(id)) {
            filteredOutput[id] = output[id];
        }
    }

    // 4. Submit partial graph to API
    // We pass 0 as the index (not used directly here) and our custom structure
    await api.queuePrompt(0, { output: filteredOutput, workflow: workflow });
};

// --- ROBUST LOCAL LOADER ---
const loadThreeJS = async () => {
    if (window._YEDP_THREE_CACHE) return window._YEDP_THREE_CACHE;

    return window._YEDP_THREE_CACHE = new Promise(async (resolve, reject) => {
        try {
            const baseUrl = new URL(".", import.meta.url).href;
            const threeUrl = new URL("three.module.js", baseUrl).href;
            const controlsUrl = new URL("OrbitControls.js", baseUrl).href;

            // Load THREE
            const THREE = await import(threeUrl);
            
            // Load OrbitControls (now that it has a valid relative import)
            const { OrbitControls } = await import(controlsUrl);

            // IMPORTANT: Removed the line causing the "not extensible" error.
            // We return the objects directly.

            resolve({ THREE, OrbitControls });
        } catch (e) {
            console.error("[Yedp] Local Loader Failed:", e);
            reject(e);
        }
    });
};

// =========================================================
// NODE 1: PREVIEW BATCH IMAGES (VIDEO PLAYER)
// =========================================================
/*app.registerExtension({
    name: "Yedp.PreviewImages",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "YedpPreviewImages") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                const self = this; 
                this.images = []; 
                this.idx = 0; 
                this.playing = false; 
                this.interval = null;

                const container = document.createElement("div");
                Object.assign(container.style, {
                    width: "100%", height: "300px", display: "flex", flexDirection: "column",
                    backgroundColor: "#000", borderRadius: "8px", overflow: "hidden", marginTop: "10px",
                    position: "relative", border: "1px solid #333"
                });

                // Status Overlay
                this.msgOverlay = document.createElement("div");
                Object.assign(this.msgOverlay.style, {
                    position: "absolute", top: 0, left: 0, width: "100%", height: "260px",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    color: "#666", fontSize: "12px", pointerEvents: "none", zIndex: "10"
                });
                this.msgOverlay.innerText = "Ready to receive images...";

                // Display Area
                this.imgDisplay = document.createElement("img");
                Object.assign(this.imgDisplay.style, { width: "100%", height: "260px", objectFit: "contain", backgroundColor: "#050505", display: "block" });
                
                // Controls
                const ctrlRow = document.createElement("div");
                Object.assign(ctrlRow.style, { display: "flex", gap: "8px", padding: "8px", background: THEME.bg, height: "40px", alignItems: "center", zIndex: "20", borderTop: "1px solid #333" });

                // LOAD BUTTON (Recovers Data Only)
                this.loadBtn = document.createElement("button"); 
                this.loadBtn.innerText = "LOAD";
                this.loadBtn.title = "Try to recover images from history, or Run path";
                Object.assign(this.loadBtn.style, { cursor: "pointer", background: THEME.btn, border: "1px solid #444", color: "#ccc", fontSize: "10px", padding: "4px 8px", borderRadius: "4px" });

                // PLAY BUTTON
                this.playBtn = document.createElement("button"); 
                this.playBtn.innerText = "▶";
                Object.assign(this.playBtn.style, { cursor: "pointer", background: "none", border: "none", color: "white", fontSize: "16px", minWidth: "20px" });
                this.playBtn.style.opacity = "0.5";

                this.slider = document.createElement("input");
                this.slider.type = "range"; this.slider.min = 0; this.slider.max = 0; this.slider.value = 0;
                this.slider.style.flex = "1";
                this.slider.style.cursor = "pointer";
                
                this.counter = document.createElement("span");
                Object.assign(this.counter.style, { fontSize: "10px", color: "#888", minWidth: "40px", textAlign: "right", fontFamily: "monospace" });
                this.counter.innerText = "0/0";

                ctrlRow.append(this.loadBtn, this.playBtn, this.slider, this.counter);
                container.append(this.msgOverlay, this.imgDisplay, ctrlRow);

                const widget = this.addDOMWidget("ui", "player", container);
                widget.computeSize = (w) => [w, 310];
                setTimeout(() => { this.setSize([300, 340]); }, 50);

                const updateFrame = (i) => {
                    // Safety: Use optional chaining to prevent crashes
                    if(!Array.isArray(self.images) || self.images.length === 0) {
                        self.msgOverlay.innerText = "No Images";
                        self.msgOverlay.style.display = "flex";
                        return;
                    }
                    self.msgOverlay.style.display = "none";
                    
                    if (i < 0) i = 0;
                    if (i >= self.images.length) i = self.images.length - 1;

                    self.idx = i;
                    const img = self.images[self.idx];
                    
                    if (img) {
                        // Safe URL construction
                        const sub = img.subfolder || "";
                        const type = img.type || "output";
                        let src = `/view?filename=${img.filename}&type=${type}&subfolder=${sub}`;
                        
                        // NOTE: For sequences, we usually DON'T want cache busting timestamps
                        // because each file is unique (001.png, 002.png). 
                        // Adding Date.now() prevents the browser from caching the frames for smooth playback.
                        
                        self.imgDisplay.src = src;
                        self.slider.value = self.idx;
                        self.counter.innerText = `${self.idx + 1}/${self.images.length}`;
                    }
                };
                
                // --- PRELOADER ---
                // Silently loads images in the background so they play smoothly
                this.preload = () => {
                    if(!self.images || self.images.length === 0) return;
                    // Cap at 200 to be safe
                    const limit = Math.min(self.images.length, 200);
                    for(let i=0; i<limit; i++) {
                        const img = self.images[i];
                        const type = img.type || "output";
                        const sub = img.subfolder || "";
                        const src = `/view?filename=${img.filename}&type=${type}&subfolder=${sub}`;
                        // Create a disconnected image object to trigger browser download
                        new Image().src = src;
                    }
                };

                // --- DATA RECOVERY LOGIC ---
                this.recoverData = () => {
                    // 1. Check ComfyUI's output cache
                    if (app.nodeOutputs && app.nodeOutputs[self.id]) {
                        const out = app.nodeOutputs[self.id];
                        // Standard node output key is "images", custom might be "yedp_images"
                        const found = out.images || out.yedp_images;
                        if (found && Array.isArray(found) && found.length > 0) {
                            console.log(`[Yedp] Recovered ${found.length} images from cache.`);
                            // CRITICAL FIX: Copy the array to prevent reference loss
                            self.images = [...found];
                            return true;
                        }
                    }
                    return false;
                };

                // Events
                this.loadBtn.onclick = async () => {
                    self.playing = false; 
                    if(self.interval) clearInterval(self.interval);
                    self.playBtn.innerText = "▶";
                    
                    if (self.recoverData()) {
                        self.slider.max = Math.max(0, self.images.length - 1);
                        self.playBtn.style.opacity = "1.0";
                        updateFrame(0);
                        self.preload(); // Trigger preload
                        self.loadBtn.innerText = "OK";
                        setTimeout(()=> self.loadBtn.innerText = "LOAD", 1500);
                    } else {
                        // Intelligent Fallback: Ask to run just this node's branch
                        if(confirm("No cache found. Queue just this node's path?")) {
                            self.loadBtn.innerText = "QUEUING...";
                            try {
                                await queueNodeAncestors(self);
                                self.loadBtn.innerText = "RUNNING";
                            } catch(e) {
                                console.error("[Yedp] Queue Failed:", e);
                                self.loadBtn.innerText = "ERR";
                            }
                        } else {
                            self.loadBtn.innerText = "EMPTY";
                        }
                        setTimeout(()=> self.loadBtn.innerText = "LOAD", 2000);
                    }
                };

                this.playBtn.onclick = () => {
                    // Auto-recover if array is empty
                    if(!self.images?.length) self.recoverData();

                    // CRITICAL FIX: Optional chaining to prevent "undefined reading length" crash
                    if(!self.images?.length) {
                        self.msgOverlay.innerText = "No Data";
                        self.msgOverlay.style.display = "flex";
                        return;
                    }

                    self.playing = !self.playing;
                    self.playBtn.innerText = self.playing ? "⏸" : "▶";
                    
                    if (self.playing) {
                        self.interval = setInterval(() => { 
                            // Safety check inside loop using optional chaining
                            if (!self.images?.length) {
                                clearInterval(self.interval);
                                self.playing = false;
                                self.playBtn.innerText = "▶";
                                return;
                            }
                            
                            let next = self.idx + 1; 
                            if (next >= self.images.length) next = 0; 
                            updateFrame(next);
                        }, 100); // 10 FPS
                    } else {
                        clearInterval(self.interval);
                    }
                };

                this.slider.oninput = (e) => {
                    self.playing = false; 
                    self.playBtn.innerText = "▶"; 
                    if (self.interval) clearInterval(self.interval);
                    updateFrame(parseInt(e.target.value));
                };
                
                this.updateFrameFunc = updateFrame;
                this.onRemoved = () => { if(self.interval) clearInterval(self.interval); };

                // Initial Recovery Attempt
                setTimeout(() => { 
                    if(self.recoverData()) {
                        self.slider.max = Math.max(0, self.images.length - 1);
                        self.playBtn.style.opacity = "1.0";
                        updateFrame(0);
                        self.preload();
                    }
                }, 1000);
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (onExecuted) onExecuted.apply(this, arguments);
                
                // Standard ComfyUI nodes send images in message.images
                const imgs = message.images || message.yedp_images;
                
                if (imgs && Array.isArray(imgs) && imgs.length > 0) {
                    // Copy data to be safe
                    this.images = [...imgs];
                    this.slider.max = Math.max(0, this.images.length - 1);
                    this.playBtn.style.opacity = "1.0";
                    this.msgOverlay.style.display = "none";
                    
                    this.idx = 0;
                    if(this.interval) clearInterval(this.interval);
                    this.playing = false;
                    this.playBtn.innerText = "▶";
                    
                    if(this.updateFrameFunc) this.updateFrameFunc(0);
                    
                    // Trigger background preload
                    if(this.preload) this.preload();
                }
            };
        }
    }
});*/

// =========================================================
// NODE 2: 3D VIEWER (THREE.JS)
// =========================================================
app.registerExtension({
    name: "Yedp.3DViewer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Yedp3DViewer") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                const self = this;
                this.poseData = []; 
                this.currentFrame = 0; 
                this.isPlaying = false; 
                this.skeletonGroup = null;

                const container = document.createElement("div");
                Object.assign(container.style, {
                    width: "100%", height: "440px", position: "relative",
                    backgroundColor: "#000", borderRadius: "8px", overflow: "hidden", marginTop: "10px",
                    border: "1px solid #333"
                });
                
                const canvasBox = document.createElement("div");
                Object.assign(canvasBox.style, { width: "100%", height: "400px", position: "relative" });
                
                const statusOverlay = document.createElement("div");
                Object.assign(statusOverlay.style, {
                    position: "absolute", top:0, left:0, width:"100%", height:"100%",
                    display: "flex", justifyContent: "center", alignItems: "center",
                    color: "#666", fontSize: "12px", pointerEvents:"none", flexDirection: "column", gap: "10px",
                    zIndex: "5"
                });
                statusOverlay.innerHTML = "<span>Initializing 3D Engine...</span>";
                canvasBox.appendChild(statusOverlay);

                const uiRow = document.createElement("div");
                Object.assign(uiRow.style, { height: "40px", display: "flex", gap: "8px", padding: "8px", background: THEME.bg, alignItems:"center", borderTop: "1px solid #333" });

                // LOAD BUTTON
                this.loadBtn = document.createElement("button"); 
                this.loadBtn.innerText = "LOAD";
                this.loadBtn.title = "Reload JSON data";
                Object.assign(this.loadBtn.style, { cursor: "pointer", background: THEME.btn, border: "1px solid #444", color: "#ccc", fontSize: "12px", padding: "4px 8px", borderRadius: "4px", marginRight: "5px" });

                this.playBtn = document.createElement("button"); this.playBtn.innerText = "▶";
                Object.assign(this.playBtn.style, { cursor: "pointer", background: "none", border: "none", color: "white", opacity: "0.5" });
                
                this.slider = document.createElement("input");
                this.slider.type = "range"; this.slider.min = 0; this.slider.max = 0; this.slider.value = 0;
                this.slider.style.flex = "1";
                this.slider.style.cursor = "pointer";

                this.frameInfo = document.createElement("span");
                Object.assign(this.frameInfo.style, { fontSize: "10px", color: "#888", fontFamily: "monospace" });
                this.frameInfo.innerText = "No Data";

                uiRow.append(this.loadBtn, this.playBtn, this.slider, this.frameInfo);
                container.append(canvasBox, uiRow);

                const widget = this.addDOMWidget("ui", "3dviewer", container);
                widget.computeSize = (w) => [w, 450]; 
                setTimeout(() => { this.setSize([400, 480]); }, 50);

                loadThreeJS().then(async ({ THREE, OrbitControls }) => {
                    statusOverlay.innerHTML = "<span>Waiting for Pose JSON...</span>"; 
                    
                    const scene = new THREE.Scene();
                    scene.background = new THREE.Color(0x050505);
                    scene.fog = new THREE.Fog(0x050505, 10, 50);

                    const camera = new THREE.PerspectiveCamera(45, canvasBox.clientWidth / canvasBox.clientHeight, 0.1, 100);
                    camera.position.set(0, 1.5, 3.5);

                    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
                    renderer.setSize(canvasBox.clientWidth, canvasBox.clientHeight);
                    renderer.setPixelRatio(window.devicePixelRatio);
                    canvasBox.appendChild(renderer.domElement);

                    const controls = new OrbitControls(camera, renderer.domElement);
                    controls.target.set(0, 1, 0);
                    controls.enableDamping = true; controls.dampingFactor = 0.05;

                    const gridHelper = new THREE.GridHelper(10, 10, 0x333333, 0x111111); scene.add(gridHelper);
                    const axesHelper = new THREE.AxesHelper(0.5); scene.add(axesHelper);
                    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 3); hemiLight.position.set(0, 20, 0); scene.add(hemiLight);
                    
                    self.skeletonGroup = new THREE.Group(); scene.add(self.skeletonGroup);

                    const mapPoint = (p, zOffset = 0) => {
                        if (!p) return new THREE.Vector3(0,0,0);
                        const x = (p.x - 0.5) * 2;
                        const y = -(p.y - 0.5) * 2 + 1.2; 
                        const z = -(p.z + zOffset) * 2; 
                        return new THREE.Vector3(x, y, z);
                    };

                    const connections = [
                        [11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28], 
                        [0,1],[1,2],[2,3],[3,4], [0,5],[5,6],[6,7],[7,8], [0,9],[9,10],[10,11],[11,12], 
                        [0,13],[13,14],[14,15],[15,16], [0,17],[17,18],[18,19],[19,20]
                    ];

                    self.updateSkeletonVisuals = (frameIdx) => {
                        if(!self.poseData || !self.poseData[frameIdx]) return;
                        
                        statusOverlay.style.display = "none";
                        const d = self.poseData[frameIdx];
                        self.skeletonGroup.clear(); 

                        const drawPts = (pts, color, scale=0.03, zOffset=0) => {
                            if(!pts) return;
                            const mat = new THREE.MeshBasicMaterial({color: color});
                            pts.forEach(p => {
                                if(!p) return;
                                const mesh = new THREE.Mesh(new THREE.SphereGeometry(scale), mat);
                                mesh.position.copy(mapPoint(p, zOffset));
                                self.skeletonGroup.add(mesh);
                            });
                        };

                        const drawLines = (pts, conns, color, zOffset=0) => {
                            if(!pts) return;
                            const mat = new THREE.LineBasicMaterial({color:color});
                            conns.forEach(([i, j]) => {
                                if(pts[i] && pts[j]) {
                                    const geo = new THREE.BufferGeometry().setFromPoints([mapPoint(pts[i], zOffset), mapPoint(pts[j], zOffset)]);
                                    const line = new THREE.Line(geo, mat);
                                    self.skeletonGroup.add(line);
                                }
                            });
                        };

                        if(d.pose) { 
                            drawPts(d.pose, 0x00ff00, 0.03); 
                            drawLines(d.pose, connections.slice(0, 12), 0xffffff); 
                        }

                        if(d.hands && d.pose) {
                            const rWrist = d.pose[16]; const lWrist = d.pose[15];
                            d.hands.forEach(h => {
                                if(!h || !h.length || !h[0]) return;
                                const root = h[0];
                                let zOffset = 0;
                                if (rWrist && lWrist) {
                                    const distR = Math.hypot(root.x - rWrist.x, root.y - rWrist.y);
                                    const distL = Math.hypot(root.x - lWrist.x, root.y - lWrist.y);
                                    zOffset = (distR < distL) ? rWrist.z : lWrist.z;
                                }
                                drawPts(h, 0x00ffff, 0.02, zOffset); 
                                drawLines(h, connections.slice(12), 0xaaaaaa, zOffset);
                            });
                        }
                        else if (d.hands) {
                             d.hands.forEach(h => { drawPts(h, 0x00ffff, 0.02); drawLines(h, connections.slice(12), 0xaaaaaa); });
                        }
                        
                        self.frameInfo.innerText = `${frameIdx + 1}/${self.poseData.length}`;
                        self.slider.value = frameIdx;
                    };

                    let lastTime = 0;
                    const animate = (time) => {
                        if(!self.isConnected) return;
                        requestAnimationFrame(animate);
                        controls.update();
                        renderer.render(scene, camera);

                        if(self.isPlaying && self.poseData && self.poseData.length > 0 && time - lastTime > 50) { 
                            lastTime = time;
                            self.currentFrame = (self.currentFrame + 1) % self.poseData.length;
                            self.updateSkeletonVisuals(self.currentFrame);
                        }
                    };
                    self.isConnected = true; animate(0);

                    self.onResize = () => {
                        setTimeout(() => {
                             if (!container || !renderer || !camera) return;
                             const w = container.clientWidth; const h = 400; 
                             renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
                        }, 50);
                    };

                }).catch(err => {
                    console.error(err);
                    statusOverlay.innerHTML = `<div style="color:#ff5555; text-align:center"><b>Local Load Failed</b><br><small>${err.message}</small></div>`;
                });

                // --- DATA RECOVERY ---
                this.recoverData = () => {
                    if (app.nodeOutputs && app.nodeOutputs[self.id]) {
                        const out = app.nodeOutputs[self.id];
                        if (out.yedp_3d_data) { 
                            // CRITICAL FIX: Copy array
                            self.poseData = [...out.yedp_3d_data]; 
                            return true; 
                        }
                    }
                    return false;
                };

                this.playBtn.onclick = () => { 
                    // Use optional chaining
                    if(!self.poseData?.length) self.recoverData();
                    if(!self.poseData?.length) return;

                    self.isPlaying = !self.isPlaying; 
                    self.playBtn.innerText = self.isPlaying ? "⏸" : "▶"; 
                };
                
                this.loadBtn.onclick = async () => {
                    self.isPlaying = false; 
                    self.playBtn.innerText = "▶";
                    
                    if (self.recoverData()) {
                        self.slider.max = Math.max(0, self.poseData.length - 1);
                        self.playBtn.style.opacity = "1.0";
                        self.currentFrame = 0; 
                        if(self.updateSkeletonVisuals) self.updateSkeletonVisuals(0);
                        self.loadBtn.innerText = "OK";
                        setTimeout(()=> self.loadBtn.innerText = "LOAD", 1500);
                    } else {
                        // Intelligent Fallback: Ask to run just this node's branch
                        if(confirm("No cache found. Queue just this node's path?")) {
                            self.loadBtn.innerText = "QUEUING...";
                            try {
                                await queueNodeAncestors(self);
                                self.loadBtn.innerText = "RUNNING";
                            } catch(e) {
                                console.error("[Yedp] Queue Failed:", e);
                                self.loadBtn.innerText = "ERR";
                            }
                        } else {
                            self.loadBtn.innerText = "EMPTY";
                        }
                        setTimeout(()=> self.loadBtn.innerText = "LOAD", 2000);
                    }
                };

                this.slider.oninput = (e) => { 
                    self.isPlaying = false; 
                    self.playBtn.innerText = "▶"; 
                    self.currentFrame = parseInt(e.target.value); 
                    if(self.updateSkeletonVisuals) self.updateSkeletonVisuals(self.currentFrame); 
                };
                
                this.onRemoved = () => { self.isConnected = false; };
                
                setTimeout(() => { 
                    if(self.recoverData()) {
                        if(self.updateSkeletonVisuals) self.updateSkeletonVisuals(0);
                    }
                }, 1000);
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (onExecuted) onExecuted.apply(this, arguments);
                if (!message || !message.yedp_3d_data) return;
                
                // Copy array
                this.poseData = [...message.yedp_3d_data];
                this.slider.max = Math.max(0, this.poseData.length - 1);
                this.playBtn.style.opacity = "1.0";
                
                this.currentFrame = 0;
                this.isPlaying = true;
                this.playBtn.innerText = "⏸";
                
                const tryDraw = () => {
                    if(this.updateSkeletonVisuals) {
                        this.updateSkeletonVisuals(0);
                    } else {
                        setTimeout(tryDraw, 100);
                    }
                };
                tryDraw();
            };
        }
    }
});
