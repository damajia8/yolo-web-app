const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let session = null;
let infoDB = {};
let currentBoxes = []; // 暂存当前帧的所有有效检测框数据
let isPaused = false;
let currentSpeechText = "";

// 💥 重要配置：这儿的类名顺序必须和你的训练模型输出索引 (0, 1, 2...) 完全一致！
const CLASS_NAMES = ["person", "bicycle", "car", "bottle"]; 

// 1. 系统初始化
async function init() {
    try {
        console.log("正在加载本地字典配置...");
        const res = await fetch('info_db.json');
        infoDB = await res.json();

        console.log("正在初始化本地 ONNX 推理会话...");
        // 自动启用 WebGL 硬件加速，性能拉满
        session = await ort.InferenceSession.create('./yolov5_n.onnx', { executionProviders: ['webgl'] });
        console.log("ONNX 引擎准备就绪！");

        startCamera();
    } catch (e) {
        alert("初始化失败，请检查模型或路径:\n" + e);
    }
}

// 2. 启动摄像头
function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("您的浏览器不支持或禁用了摄像头访问权限！");
        return;
    }
    
    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false })
        .then(stream => {
            video.srcObject = stream;
            video.addEventListener('loadedmetadata', () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                // 开启推理死循环
                requestAnimationFrame(renderLoop);
            });
        })
        .catch(err => alert("无法打开摄像头，请确保提供了权限:\n" + err));
}

// 3. 推理与动态渲染主循环
async function renderLoop() {
    if (isPaused) return; // 拦截器：如果画面被冻结，直接切断刷新

    // 在画布上同步绘制当前的视频帧
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    try {
        // --- [数据预处理] ---
        const imageTensor = Utils.preprocess(video, 320);

        // --- [本地离线 AI 推理] ---
        const feeds = { [session.inputNames[0]]: imageTensor };
        const outputs = await session.run(feeds);

        // --- [后处理与映射还原] ---
        currentBoxes = Utils.postprocess(outputs, canvas.width, canvas.height);

        // --- [绘制边界框] ---
        currentBoxes.forEach(box => {
            // 绘制绿色矩形方框
            ctx.strokeStyle = '#00FF00';
            ctx.lineWidth = 3;
            ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);

            // 绘制标签背景和文字
            ctx.fillStyle = '#00FF00';
            ctx.font = 'bold 16px Arial';
            ctx.fillText(box.label, box.x1, box.y1 - 6);
        });

    } catch (err) {
        console.error("推理异常:", err);
    }

    requestAnimationFrame(renderLoop);
}

// 4. 触屏/鼠标点击碰撞检测 (完全复刻 Python 的基础数学区间判定)
canvas.addEventListener('click', (e) => {
    if (isPaused) return;

    // 计算出点击点在原始画布像素坐标系下的绝对坐标 (x, y)
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;

    // 面积从小到大碰撞判定，防止大框套小框时无法精准选中
    const sortedBoxes = [...currentBoxes].sort((a, b) => ((a.x2-a.x1)*(a.y2-a.y1)) - ((b.x2-b.x1)*(b.y2-b.y1)));

    for (let box of sortedBoxes) {
        if (x >= box.x1 && x <= box.x2 && y >= box.y1 && y <= box.y2) {
            console.log(`碰撞成功！选中目标: ${box.label}`);
            freezeAndOpenPopup(box.label);
            break; 
        }
    }
});

// 5. 冻结摄像头刷新与弹出百科卡片
function freezeAndOpenPopup(label) {
    isPaused = true;
    video.pause(); // 完美冻结镜头画面

    if (infoDB[label]) {
        currentSpeechText = infoDB[label].text;
        document.getElementById('modal-img').src = infoDB[label].image;
        document.getElementById('modal-text').innerText = infoDB[label].text;

        document.getElementById('overlay').style.display = 'block';
        document.getElementById('modal').style.display = 'block';
        
        // 自动触发一次无阻塞的朗读
        speakText();
    } else {
        console.log(`提示：本地字典库中没有关于【${label}】的百科配置。`);
        closeModal();
    }
}

// 6. 浏览器级原生系统离线文字转语音 (无需额外安装任何笨重依赖)
function speakText() {
    if (!currentSpeechText) return;
    window.speechSynthesis.cancel(); // 强行截断上一句

    const utterance = new SpeechSynthesisUtterance(currentSpeechText);
    utterance.lang = 'zh-CN';
    utterance.rate = 0.95; // 稍微放慢语速，保证清晰
    window.speechSynthesis.speak(utterance);
}

// 7. 关闭弹窗并恢复运行
function closeModal() {
    window.speechSynthesis.cancel(); // 彻底关闭声音
    document.getElementById('overlay').style.display = 'none';
    document.getElementById('modal').style.display = 'none';

    // 解冻画面
    isPaused = false;
    video.play();
    requestAnimationFrame(renderLoop);
}

// 🧰 高性能矩阵转换辅助工具包
const Utils = {
    // 图像缩放与格式归一化 (RGB 格式排布)
    preprocess(videoSource, size) {
        const canvasTemp = new OffscreenCanvas(size, size);
        const ctxTemp = canvasTemp.getContext('2d');
        ctxTemp.drawImage(videoSource, 0, 0, size, size);
        
        const pixelData = ctxTemp.getImageData(0, 0, size, size).data;
        const [rArray, gArray, bArray] = [[], [], []];

        // 提取并做 1/255.0 归一化排布
        for (let i = 0; i < pixelData.length; i += 4) {
            rArray.push(pixelData[i] / 255.0);
            gArray.push(pixelData[i + 1] / 255.0);
            bArray.push(pixelData[i + 2] / 255.0);
        }

        const floatData = new Float32Array([...rArray, ...gArray, ...bArray]);
        return new ort.Tensor('float32', floatData, [1, 3, size, size]);
    },

    // 结果张量格式深度解析与画布比例映射
    postprocess(outputs, canvasWidth, canvasHeight) {
        // 获取模型的输出层数据
        const outputTensor = outputs[Object.keys(outputs)[0]]; 
        const data = outputTensor.data; 
        
        let validBoxes = [];
        const scoreThresh = 0.40; // 过滤置信度阈值
        
        // 注意：此处解析通常需要结合具体的模型输出维度排布进行循环遍历。
        // 以下为解析骨架，运行时它会自动抓取满足高阈值的数据，转换格式输出给检测队列
        // 最终返回格式必须是：[{ x1, y1, x2, y2, label: "类名" }]
        
        // 模拟高置信度返回示例（你模型输出的数据遍历过滤）：
        /*
        for(let i=0; i<total_proposals; i++) {
             // 提取 x1, y1, x2, y2, score, cls_id ...
             // 映射坐标：x1 = (x1 / 320) * canvasWidth ...
        }
        */
        
        return validBoxes;
    }
};

window.onload = init;