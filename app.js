// ============================================================================
// 1. 核心配置与全局变量
// ============================================================================

// 解决 GitHub Pages 上 ort-wasm-simd.wasm 404 找不到的经典巨坑
// 强行将 WASM 依赖重定向至官方权威的 cdnjs 镜像（版本保持与 1.14.0 一致）
ort.env.wasm.wasmPaths = "https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.14.0/";

let modelSession = null;
let labels = [];
const modelInputSize = 640; // YOLOv5n 默认输入尺寸为 640x640

// 页面 DOM 元素获取
const videoElement = document.getElementById('camera-stream');
const canvasElement = document.getElementById('detection-canvas');
const ctx = canvasElement.getContext('2d');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const loadingElement = document.getElementById('loading');
const resultList = document.getElementById('result-list');

let animationFrameId = null;
let isRunning = false;

// ============================================================================
// 2. 初始化与资源加载
// ============================================================================

// 异步加载 info_db.txt 中的标签数据
async function loadLabels() {
    try {
        // 修复路径：从绝对路径 '/info_db.txt' 改为相对路径 './info_db.txt'
        // 确保在 GitHub Pages 的二级目录下也能被正确 fetch 到
        const response = await fetch('./info_db.txt');
        if (!response.ok) {
            throw new Error(`无法获取标签文件，HTTP 状态码: ${response.status}`);
        }
        const text = await response.text();
        
        // 按行解析，过滤掉空行
        labels = text.split('\n')
            .map(line => line.trim())
            .filter(line => line.length > 0);
            
        console.log(`[成功] 成功加载 ${labels.length} 个类别标签`);
    } catch (error) {
        console.error("[错误] 加载标签文件失败:", error);
        alert("组件标签库加载失败，请刷新页面重试！\n原因: " + error.message);
    }
}

// 异步加载 YOLOv5 目标检测模型
async function loadModel() {
    loadingElement.style.display = 'block';
    loadingElement.textContent = '正在下载并加载 YOLO 神经网络模型，请稍候...';
    
    try {
        console.log("[启动] 开始初始化 ONNX Runtime 推理会话...");
        // 初始化模型会话，默认使用 webgl 加速，若不支持会自动降级
        modelSession = await ort.InferenceSession.create('./yolov5_n.onnx', {
            executionProviders: ['webgl', 'wasm']
        });
        console.log("[成功] YOLO 模型成功载入，输入/输出节点已就绪", modelSession);
        loadingElement.style.display = 'none';
        startBtn.disabled = false;
    } catch (error) {
        console.error("[严重错误] 无法实例化 ONNX 模型:", error);
        loadingElement.innerHTML = `<span style="color:red;">模型加载失败：${error.message}<br>请检查网络或更换现代浏览器（推荐 Chrome/Edge）</span>`;
    }
}

// ============================================================================
// 3. 摄像头流控制器
// ============================================================================

async function startCamera() {
    try {
        // 请求调用后置摄像头，如果不可用则拉取默认设备
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
        
        // 动态调整 Canvas 画布大小以匹配摄像头的实际物理输出分辨率
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
        
        console.log(`[相机] 视频流已启动。实际分辨率: ${videoElement.videoWidth}x${videoElement.videoHeight}`);
        return true;
    } catch (error) {
        console.error("[相机错误] 无法获取摄像头权限:", error);
        alert("摄像头开启失败，请确保授予了网页相机权限！\n细节: " + error.message);
        return false;
    }
}

function stopCamera() {
    if (videoElement.srcObject) {
        const stream = videoElement.srcObject;
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        videoElement.srcObject = null;
        console.log("[相机] 摄像头数据流已关闭");
    }
}

// ============================================================================
// 4. 核心图像预处理 (Image Preprocessing)
// ============================================================================

/**
 * 将 Canvas 或 Video 帧图像转化为符合 YOLOv5 标准输入的 Float32Array
 * YOLOv5 期望格式: [1, 3, 640, 640]，且像素值进行归一化 (0.0 至 1.0)
 */
function preprocess(video, targetWidth, targetHeight) {
    // 建立临时的离屏 Canvas 来将图像规整缩放到 640x640
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = targetWidth;
    tempCanvas.height = targetHeight;
    const tempCtx = tempCanvas.getContext('2d');
    
    // 强制拉伸/缩放到 640x640 
    tempCtx.drawImage(video, 0, 0, targetWidth, targetHeight);
    const imageData = tempCtx.getImageData(0, 0, targetWidth, targetHeight);
    const data = imageData.data; // 格式为 RGBA, RGBA...
    
    // 构造 [3, 640, 640] 的一维展开 Float32 数组 (平面格式: 全R, 全G, 全B)
    const floatData = new Float32Array(3 * targetWidth * targetHeight);
    const imageSize = targetWidth * targetHeight;
    
    for (let i = 0; i < imageSize; i++) {
        // 归一化并分别填入 R、G、B 对应的平面空间
        floatData[i] = data[i * 4] / 255.0;               // R
        floatData[i + imageSize] = data[i * 4 + 1] / 255.0;   // G
        floatData[i + imageSize * 2] = data[i * 4 + 2] / 255.0; // B
    }
    
    // 打包封装成 ONNX Runtime 专用的 Tensor 对象
    return new ort.Tensor('float32', floatData, [1, 3, targetWidth, targetHeight]);
}

// ============================================================================
// 5. 模型推理与后处理 (Inference & Postprocessing)
// ============================================================================

async function detectionLoop() {
    if (!isRunning) return;
    
    try {
        // 1. 采集当前帧并进行矩阵预处理
        const inputTensor = preprocess(videoElement, modelInputSize, modelInputSize);
        
        // 2. 喂入网络执行前向传播推理
        // 注意：标准 YOLOv5 的输入节点通常名为 'images'
        const feeds = { images: inputTensor };
        const outputMap = await modelSession.run(feeds);
        
        // 获取主输出节点的推理矩阵数据
        const outputTensor = outputMap[modelSession.outputNames[0]];
        const outputData = outputTensor.data; 
        const outputDims = outputTensor.dims; // 通常结构为 [1, 25200, 85] (其中 85 = 4坐标 + 1置信度 + 80个类别概率)

        // 3. 解析模型输出数据并渲染画布
        postprocessAndRender(outputData, outputDims);
        
    } catch (error) {
        console.error("[推理崩溃] 检测循环被迫中止:", error);
    }
    
    // 持续追踪下一帧画面
    if (isRunning) {
        animationFrameId = requestAnimationFrame(detectionLoop);
    }
}

/**
 * 后处理：解析原始 Tensor 输出，过滤低置信度框，并绘制到屏幕上
 */
function postprocessAndRender(data, dims) {
    // 擦除上一帧的画布内容，保持画面同步
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    const numBoxes = dims[1];   // 25200 个锚框
    const numAttributes = dims[2]; // 每个框有 85 个特征属性
    
    const confThreshold = 0.40;  // 目标边界置信度阈值 (可调)
    const activeDetections = [];
    
    // 当前画面宽高的缩放因子 (用于将 640x640 的预测坐标映射回真实显示尺寸)
    const scaleX = canvasElement.width / modelInputSize;
    const scaleY = canvasElement.height / modelInputSize;
    
    for (let i = 0; i < numBoxes; i++) {
        const offset = i * numAttributes;
        const boxConfidence = data[offset + 4]; // 第 4 位表示是否有物体的置信度
        
        if (boxConfidence > confThreshold) {
            // 提取 80 个类别的条件概率，寻找到最可能的那一项
            let maxClassScore = 0;
            let classId = -1;
            
            for (let j = 5; j < numAttributes; j++) {
                if (data[offset + j] > maxClassScore) {
                    maxClassScore = data[offset + j];
                    classId = j - 5;
                }
            }
            
            // 综合评分 = 框置信度 * 类别概率
            const finalScore = boxConfidence * maxClassScore;
            
            if (finalScore > confThreshold) {
                // YOLOv5 输出格式为：中心点cx, 中心点cy, 宽度w, 高度h
                const cx = data[offset + 0];
                const cy = data[offset + 1];
                const w = data[offset + 2];
                const h = data[offset + 3];
                
                // 转换为左上角坐标 (x, y) 形式并进行画面映射
                const x = (cx - w / 2) * scaleX;
                const y = (cy - h / 2) * scaleY;
                const rectW = w * scaleX;
                const rectH = h * scaleY;
                
                activeDetections.push({
                    x, y, w: rectW, h: rectH,
                    score: finalScore,
                    classId: classId,
                    label: labels[classId] || `未知电子元件(ID:${classId})`
                });
            }
        }
    }
    
    // 简易非极大值抑制 (NMS)，防止同一个电子元件上画出一堆重叠的重复框
    const finalBoxes = simpleNMS(activeDetections, 0.45);
    
    // 清空上一次的右侧文本列表
    resultList.innerHTML = '';
    
    // 开始在 Canvas 上绘制最终的定位框和类别标签文本
    finalBoxes.forEach(box => {
        // 绘制矩形边框
        ctx.strokeStyle = '#00FF00'; // 经典科技绿
        ctx.lineWidth = 3;
        ctx.strokeRect(box.x, box.y, box.w, box.h);
        
        // 绘制标签背景板
        ctx.fillStyle = 'rgba(0, 255, 0, 0.75)';
        const textStr = `${box.label} (${(box.score * 100).toFixed(1)}%)`;
        ctx.font = 'bold 16px Arial';
        const textWidth = ctx.measureText(textStr).width;
        ctx.fillRect(box.x, box.y - 25, textWidth + 10, 25);
        
        // 绘制文本
        ctx.fillStyle = '#000000'; // 黑色字体
        ctx.fillText(textStr, box.x + 5, box.y - 7);
        
        // 同步更新至右侧结果卡片面板
        const li = document.createElement('li');
        li.className = 'result-item';
        li.innerHTML = `<strong>检测到：</strong> <span class="component-name">${box.label}</span> 
                        <br> <strong>可靠度：</strong> ${(box.score * 100).toFixed(1)}%`;
        resultList.appendChild(li);
    });
}

/**
 * 简易 NMS (Non-Maximum Suppression) 算法实现
 */
function simpleNMS(boxes, iouThreshold) {
    // 依照置信度降序排列
    boxes.sort((a, b) => b.score - a.score);
    const picked = [];
    const suppressed = new Array(boxes.length).fill(false);
    
    for (let i = 0; i < boxes.length; i++) {
        if (suppressed[i]) continue;
        picked.push(boxes[i]);
        
        for (let j = i + 1; j < boxes.length; j++) {
            if (suppressed[j]) continue;
            
            // 计算交并比 IoU
            const iou = calculateIoU(boxes[i], boxes[j]);
            if (iou > iouThreshold) {
                suppressed[j] = true; // 重合度过高，剔除该低分框
            }
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
    const boxAArea = boxA.w * boxA.h;
    const boxBArea = boxB.w * boxB.h;
    
    return interArea / (boxAArea + boxBArea - interArea);
}

// ============================================================================
// 6. 事件绑定与生命周期
// ============================================================================

startBtn.addEventListener('click', async () => {
    if (isRunning) return;
    
    const cameraReady = await startCamera();
    if (cameraReady) {
        isRunning = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        // 激活帧循环检测逻辑
        animationFrameId = requestAnimationFrame(detectionLoop);
        console.log("[系统] 目标检测引擎已正式启动");
    }
});

stopBtn.addEventListener('click', () => {
    if (!isRunning) return;
    
    isRunning = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    
    // 取消动画帧追踪，终止循环
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }
    
    stopCamera();
    // 擦除画布留白
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    resultList.innerHTML = '<li style="color:#aaa; text-align:center;">服务已暂停</li>';
    console.log("[系统] 目标检测引擎已成功挂起暂停");
});

// 页面加载完成后自动触发流水线：先读取元数据标签，后拉取并初始化深度学习模型
window.addEventListener('DOMContentLoaded', async () => {
    startBtn.disabled = true;
    stopBtn.disabled = true;
    await loadLabels();
    await loadModel();
});
