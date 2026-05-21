const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const thinkingMode = document.getElementById('thinking-mode');
const thinkingText = document.getElementById('thinking-text');

let session = null;
let infoDB = {};
let currentBoxes = []; 
let isPaused = false;
let currentSpeechText = "";
let audioUnlocked = false; // iOS 语音解锁标记

let lastFrameTime = 0;
const FPS_LIMIT = 8; 
const MODEL_SIZE = 320;
const offscreenCanvas = new OffscreenCanvas(MODEL_SIZE, MODEL_SIZE);
const offscreenCtx = offscreenCanvas.getContext('2d');

const CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", 
    "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]; 

// iOS/Safari 语音强制解锁逻辑
document.body.addEventListener('touchstart', () => {
    if (!audioUnlocked) {
        const dummyUtterance = new SpeechSynthesisUtterance('');
        window.speechSynthesis.speak(dummyUtterance);
        audioUnlocked = true;
    }
}, { once: true });

async function init() {
    try {
        thinkingText.innerText = "加载离线数据库...";
        const res = await fetch('info_db.json');
        infoDB = await res.json();
        
        thinkingText.innerText = "初始化 AI 模型 (首次需几秒)...";
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        session = await ort.InferenceSession.create('./yolov5_n.onnx', { executionProviders: ['wasm'] });
        
        startCamera();
    } catch (e) {
        thinkingText.innerText = "初始化失败: " + e.message;
    }
}

function startCamera() {
    thinkingText.innerText = "启动摄像头...";
    // 强制限制分辨率，保护手机性能
    const constraints = {
        video: { 
            facingMode: 'environment',
            width: { ideal: 640 },
            height: { ideal: 480 }
        }, 
        audio: false 
    };

    navigator.mediaDevices.getUserMedia(constraints)
        .then(stream => {
            video.srcObject = stream;
            video.addEventListener('loadedmetadata', () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                thinkingMode.style.display = 'none'; // 隐藏思考 UI
                requestAnimationFrame(renderLoop);
            });
        }).catch(err => {
            thinkingText.innerText = "无法打开摄像头: " + err;
        });
}

async function renderLoop(timestamp) {
    if (isPaused) return;

    if (timestamp - lastFrameTime < (1000 / FPS_LIMIT)) {
        requestAnimationFrame(renderLoop);
        return;
    }
    lastFrameTime = timestamp;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    try {
        const imageTensor = Utils.preprocess(video, MODEL_SIZE);
        const feeds = { [session.inputNames[0]]: imageTensor };
        const outputs = await session.run(feeds);

        currentBoxes = Utils.postprocess(outputs, canvas.width, canvas.height);

        currentBoxes.forEach(box => {
            ctx.strokeStyle = '#00FF00';
            ctx.lineWidth = 3;
            ctx.strokeRect(box.x1, box.y1, box.w, box.h);

            ctx.fillStyle = 'rgba(0, 255, 0, 0.7)';
            ctx.fillRect(box.x1, box.y1 - 25, ctx.measureText(box.label).width + 10, 25);
            ctx.fillStyle = '#000000';
            ctx.font = 'bold 16px Arial';
            ctx.fillText(box.label, box.x1 + 5, box.y1 - 6);
        });
    } catch (err) {
        console.error("推理异常:", err);
    }
    requestAnimationFrame(renderLoop);
}

canvas.addEventListener('click', (e) => {
    if (isPaused) return;
    const rect = canvas.getBoundingClientRect();
    // 修正 Canvas 缩放导致的坐标偏移
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    const sortedBoxes = [...currentBoxes].sort((a, b) => (a.w * a.h) - (b.w * b.h));
    for (let box of sortedBoxes) {
        if (x >= box.x1 && x <= box.x1 + box.w && y >= box.y1 && y <= box.y1 + box.h) {
            freezeAndOpenPopup(box.label);
            break; 
        }
    }
});

function freezeAndOpenPopup(label) {
    isPaused = true;
    video.pause(); 
    if (infoDB[label]) {
        currentSpeechText = infoDB[label].text;
        document.getElementById('modal-img').src = infoDB[label].image;
        document.getElementById('modal-text').innerText = infoDB[label].text;
        document.getElementById('overlay').style.display = 'block';
        document.getElementById('modal').style.display = 'block';
        speakText();
    } else {
        closeModal();
    }
}

function speakText() {
    if (!currentSpeechText) return;
    window.speechSynthesis.cancel(); 
    const utterance = new SpeechSynthesisUtterance(currentSpeechText);
    utterance.lang = 'zh-CN';
    utterance.rate = 0.95; 
    window.speechSynthesis.speak(utterance);
}

function closeModal() {
    window.speechSynthesis.cancel(); 
    document.getElementById('overlay').style.display = 'none';
    document.getElementById('modal').style.display = 'none';
    isPaused = false;
    video.play();
    requestAnimationFrame(renderLoop);
}

const Utils = {
    preprocess(videoSource, size) {
        offscreenCtx.drawImage(videoSource, 0, 0, size, size);
        const pixelData = offscreenCtx.getImageData(0, 0, size, size).data;
        const floatData = new Float32Array(3 * size * size);
        
        for (let i = 0, j = 0; i < pixelData.length; i += 4, j++) {
            floatData[j] = pixelData[i] / 255.0;
            floatData[j + size * size] = pixelData[i + 1] / 255.0;
            floatData[j + 2 * size * size] = pixelData[i + 2] / 255.0;
        }
        return new ort.Tensor('float32', floatData, [1, 3, size, size]);
    },
    postprocess(outputs, canvasWidth, canvasHeight) {
        let validBoxes = [];
        const CONF_THRESH = 0.45;
        const scaleX = canvasWidth / MODEL_SIZE;
        const scaleY = canvasHeight / MODEL_SIZE;

        for (const key in outputs) {
            const data = outputs[key].data;
            const dims = outputs[key].dims; 
            
            if (dims.length === 3) {
                const numProposals = dims[1];
                const numAttrs = dims[2];
                for (let i = 0; i < numProposals; i++) {
                    const offset = i * numAttrs;
                    const conf = data[offset + 4];
                    if (conf < CONF_THRESH) continue;

                    let maxClassProb = 0;
                    let classId = -1;
                    for (let c = 0; c < CLASS_NAMES.length; c++) {
                        const prob = data[offset + 5 + c];
                        if (prob > maxClassProb) { maxClassProb = prob; classId = c; }
                    }

                    const score = conf * maxClassProb;
                    if (score > CONF_THRESH) {
                        const cx = data[offset];
                        const cy = data[offset + 1];
                        const w = data[offset + 2];
                        const h = data[offset + 3];
                        
                        validBoxes.push({
                            x1: (cx - w / 2) * scaleX,
                            y1: (cy - h / 2) * scaleY,
                            w: w * scaleX,
                            h: h * scaleY,
                            score: score,
                            label: CLASS_NAMES[classId]
                        });
                    }
                }
            }
        }
        return this.nms(validBoxes, 0.45);
    },
    nms(boxes, iouThresh) {
        boxes.sort((a, b) => b.score - a.score);
        const result = [];
        while (boxes.length > 0) {
            const bestBox = boxes.shift();
            result.push(bestBox);
            boxes = boxes.filter(box => this.iou(bestBox, box) < iouThresh);
        }
        return result;
    },
    iou(box1, box2) {
        const xA = Math.max(box1.x1, box2.x1);
        const yA = Math.max(box1.y1, box2.y1);
        const xB = Math.min(box1.x1 + box1.w, box2.x1 + box2.w);
        const yB = Math.min(box1.y1 + box1.h, box2.y1 + box2.h);
        const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
        const box1Area = box1.w * box1.h;
        const box2Area = box2.w * box2.h;
        return interArea / (box1Area + box2Area - interArea);
    }
};

window.onload = init;
