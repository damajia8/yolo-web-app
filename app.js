// ============================================================================
// 1. 核心配置与全局变量
// ============================================================================

// 解决手机端/GitHub Pages 上 WASM 文件 404 及加载失败的巨坑
ort.env.wasm.wasmPaths = "https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.14.0/";

let modelSession = null;
let labels = [];
const modelInputSize = 640; // YOLOv5n 默认输入尺寸

// 页面 DOM 元素
const videoElement = document.getElementById('camera-stream');
const canvasElement = document.getElementById('detection-canvas');
const ctx = canvasElement.getContext('2d');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const loadingElement = document.getElementById('loading');
const resultList = document.getElementById('result-list');

let animationFrameId = null;
let isRunning = false;
let currentDetections = []; // 存储当前帧检测到的所有框，用于点击判定

// ============================================================================
// 2. 初始化与资源加载
// ============================================================================

// 异步加载标签数据
async function loadLabels() {
    try {
        const response = await fetch('./info_db.txt');
        if (!response.ok) throw new Error(`HTTP 状态码: ${response.status}`);
        const text = await response.text();
        labels = text.split('\n').map(line => line.trim()).filter(line => line.length > 0);
        console.log(`[成功] 加载了 ${labels.length} 个标签`);
    } catch (error) {
        console.error("加载标签失败:", error);
    }
}

// 异步加载 YOLOv5 模型（手机端强力兼容版）
async function loadModel() {
    loadingElement.style.display = 'block';
    loadingElement.textContent = '神经网络正载入手机内存，请稍候...';
    
    // 策略一：尝试 WebGL
    try {
        console.log("尝试使用 WebGL 模式...");
        modelSession = await ort.InferenceSession.create('./yolov5_n.onnx', { executionProviders: ['webgl'] });
        console.log("WebGL 初始化成功");
        loadingElement.style.display = 'none';
        startBtn.disabled = false;
        return;
    } catch (e) {
        console.warn("手机不支持 WebGL，准备切换至 WASM 兼容模式...", e);
    }

    // 策略二：绝对兼容的 WASM 纯 CPU 模式
    try {
        console.log("尝试使用 WASM 模式...");
        modelSession = await ort.InferenceSession.create('./yolov5_n.onnx', { executionProviders: ['wasm'] });
        console.log("WASM 初始化成功");
        loadingElement.style.display = 'none';
        startBtn.disabled = false;
    } catch (error) {
        console.error("所有推理引擎均在手机端崩溃:", error);
        loadingElement.innerHTML = `<span style="color:red;">手机加载失败: ${error.message}</span>`;
    }
}

// ============================================================================
// 3. 摄像头流控制器
// ============================================================================

async function startCamera() {
    try {
        // 强制调用手机后置环境摄像头
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment', 
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });
        
        videoElement.srcObject = stream;
        await videoElement.play();
        
        // 让 Canvas 的画布分辨率与手机获取到的摄像头像素完全对齐
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
        
        return true;
    } catch (error) {
        alert("手机相机权限获取失败，请确保使用 HTTPS 访问并允许了相机权限！");
        return false;
    }
}

function stopCamera() {
    if (videoElement.srcObject) {
        videoElement.srcObject.getTracks().forEach(track => track.stop());
        videoElement.srcObject = null;
    }
}

// ============================================================================
// 4. 图像预处理与推理
// ============================================================================

function preprocess(video, targetWidth, targetHeight) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = targetWidth;
    tempCanvas.height = targetHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0, targetWidth, targetHeight);
    
    const data = tempCtx.getImageData(0, 0, targetWidth, targetHeight).data;
    const floatData = new Float32Array(3 * targetWidth * targetHeight);
    const imageSize = targetWidth * targetHeight;
    
    for (let i = 0; i < imageSize; i++) {
        floatData[i] = data[i * 4] / 255.0;               // R
        floatData[i + imageSize] = data[i * 4 + 1] / 255.0;   // G
        floatData[i + imageSize * 2] = data[i * 4 + 2] / 255.0; // B
    }
    return new ort.Tensor('float32', floatData, [1, 3, targetWidth, targetHeight]);
}

async function detectionLoop() {
    if (!isRunning) return;
    try {
        const inputTensor = preprocess(videoElement, modelInputSize, modelInputSize);
        const outputMap = await modelSession.run({ images: inputTensor });
        const outputTensor = outputMap[modelSession.outputNames[0]];
        
        postprocessAndRender(outputTensor.data, outputTensor.dims);
    } catch (error) {
        console.error("帧处理失败:", error);
    }
    if (isRunning) {
        animationFrameId = requestAnimationFrame(detectionLoop);
    }
}

// ============================================================================
// 5. 后处理与画框渲染（包含手机点击事件支持）
// ============================================================================

function postprocessAndRender(data, dims) {
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    const numBoxes = dims[1];
    const numAttributes = dims[2];
    const confThreshold = 0.35; // 置信度阈值
    const activeDetections = [];
    
    const scaleX = canvasElement.width / modelInputSize;
    const scaleY = canvasElement.height / modelInputSize;
    
    for (let i = 0; i < numBoxes; i++) {
        const offset = i * numAttributes;
        const boxConfidence = data[offset + 4];
        
        if (boxConfidence > confThreshold) {
            let maxClassScore = 0;
            let classId = -1;
            for (let j = 5; j < numAttributes; j++) {
                if (data[offset + j] > maxClassScore) {
                    maxClassScore = data[offset + j];
                    classId = j - 5;
                }
            }
            
            const finalScore = boxConfidence * maxClassScore;
            if (finalScore > confThreshold) {
                const cx = data[offset + 0];
                const cy = data[offset + 1];
                const w = data[offset + 2];
                const h = data[offset + 3];
                
                const x = (cx - w / 2) * scaleX;
                const y = (cy - h / 2) * scaleY;
                const rectW = w * scaleX;
                const rectH = h * scaleY;
                
                activeDetections.push({
                    x, y, w: rectW, h: rectH,
                    score: finalScore,
                    classId: classId,
                    label: labels[classId] || `元件(ID:${classId})`
                });
            }
        }
    }
    
    // 执行 NMS 并保存到全局变量中供点击检测使用
    currentDetections = simpleNMS(activeDetections, 0.45);
    resultList.innerHTML = '';
    
    currentDetections.forEach(box => {
        // 1. 绘制矩形选框
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 4;
        ctx.strokeRect(box.x, box.y, box.w, box.h);
        
        // 2. 绘制交互标签头（增大手机端触控视觉面积）
        const textStr = `👉 ${box.label} [点击了解]`;
        ctx.font = 'bold 18px Arial';
        const textWidth = ctx.measureText(textStr).width;
        
        ctx.fillStyle = 'rgba(0, 255, 0, 0.85)';
        // 存储标签的理论物理点击区域 (用于判定点击事件)
        box.labelArea = { x: box.x, y: box.y - 32, w: textWidth + 16, h: 32 };
        ctx.fillRect(box.labelArea.x, box.labelArea.y, box.labelArea.w, box.labelArea.h);
        
        // 3. 填入文字
        ctx.fillStyle = '#000000';
        ctx.fillText(textStr, box.x + 8, box.y - 10);
        
        // 4. 同步生成下方的文本结果卡片（手机点击这个卡片也能跳转）
        const li = document.createElement('li');
        li.className = 'result-item';
        li.style.borderLeft = "5px solid #00FF00";
        li.style.padding = "8px";
        li.style.marginBottom = "5px";
        li.style.backgroundColor = "#f0f0f0";
        li.innerHTML = `<strong>名称：</strong> <span style="color:#008c00;">${box.label}</span> (点击跳转介绍)`;
        // 为下方的列表项绑定相同的跳转
        li.onclick = () => handleLabelRedirect(box.label);
        resultList.appendChild(li);
    });
}

// 统一的跳转处理函数
function handleLabelRedirect(labelName) {
    // 停止检测循环，防止弹窗或跳转后手机端卡死
    stopBtn.click();
    
    // 根据识别到的标签名字，拼接你想要的跳转目标（如百度百科、国内电子元件库网等）
    // 你可以把这里的 URL 改成你任何想去的地方
    const targetUrl = `https://baike.baidu.com/item/${encodeURIComponent(labelName)}`;
    
    alert(`即将离开本应用，前往了解：${labelName}`);
    window.location.href = targetUrl;
}

// 监听手机屏幕上的 Canvas 点击（触摸）事件
canvasElement.addEventListener('touchstart', (event) => {
    // 阻止默认的缩放行为
    event.preventDefault();
    
    // 获取手机触屏相对 Canvas 画布的真实物理坐标
    const rect = canvasElement.getBoundingClientRect();
    const touch = event.touches[0];
    const clickX = ((touch.clientX - rect.left) / rect.width) * canvasElement.width;
    const clickY = ((touch.clientY - rect.top) / rect.height) * canvasElement.height;
    
    // 遍历当前帧的所有检测框，判断手指点中了哪一个
    for (let box of currentDetections) {
        // 情况一：点中了绿色的标签文字区域
        if (box.labelArea && 
            clickX >= box.labelArea.x && clickX <= box.labelArea.x + box.labelArea.w &&
            clickY >= box.labelArea.y && clickY <= box.labelArea.y + box.labelArea.h) {
            handleLabelRedirect(box.label);
            return;
        }
        // 情况二：直接点中了识别框内部
        if (clickX >= box.x && clickX <= box.x + box.w &&
            clickY >= box.y && clickY <= box.y + box.h) {
            handleLabelRedirect(box.label);
            return;
        }
    }
}, { passive: false });

// NMS 算法核心
function simpleNMS(boxes, iouThreshold) {
    boxes.sort((a, b) => b.score - a.score);
    const picked = [];
    const suppressed = new Array(boxes.length).fill(false);
    for (let i = 0; i < boxes.length; i++) {
        if (suppressed[i]) continue;
        picked.push(boxes[i]);
        for (let j = i + 1; j < boxes.length; j++) {
            if (suppressed[j]) continue;
            if (calculateIoU(boxes[i], boxes[j]) > iouThreshold) suppressed[j] = true;
        }
    }
    return picked;
}

function calculateIoU(boxA, boxB) {
    const xA = Math.max(boxA.x, boxB.x);
    const yA = Math.max(boxA.y, boxB.y);
    const xB = Math.min(boxA.x + boxA.w, boxB.x + boxB.w);
    const yB = Math.min(boxA.y + boxA.h, boxB.y + boxB.h);
    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    return interArea / (boxA.w * boxA.h + boxB.w * boxB.h - interArea);
}

// ============================================================================
// 6. 生命周期绑定
// ============================================================================

startBtn.addEventListener('click', async () => {
    if (isRunning) return;
    if (await startCamera()) {
        isRunning = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        animationFrameId = requestAnimationFrame(detectionLoop);
    }
});

stopBtn.addEventListener('click', () => {
    if (!isRunning) return;
    isRunning = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    if (animationFrameId) cancelAnimationFrame(animationFrameId);
    stopCamera();
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    resultList.innerHTML = '<li style="color:#aaa;text-align:center;">服务已暂停</li>';
});

window.addEventListener('DOMContentLoaded', async () => {
    startBtn.disabled = true;
    stopBtn.disabled = true;
    await loadLabels();
    await loadModel();
});
