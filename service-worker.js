const CACHE_NAME = 'yolo-百科离线应用-v1';

// 需要被强制打包并离线锁死到手机本地的文件清单
const STATIC_ASSETS = [
    './',
    './index.html',
    './app.js',
    './info_db.json',
    './yolov5_n.onnx',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js',
    './images/person.jpg',
    './images/bottle.jpg'
];

// 安装阶段：把上面清单里的所有大文件（包括ONNX模型、文字、网页、图片）全部拉取存进闪存
self.addEventListener('install', event => {
    console.log('[Service Worker] 正在执行安装，缓存数据中...');
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => {
            return cache.addAll(STATIC_ASSETS);
        }).then(() => self.skipWaiting())
    );
});

// 激活阶段：清理旧版本的多余缓存
self.addEventListener('activate', event => {
    console.log('[Service Worker] 激活成功，正在接管控制权...');
    event.waitUntil(
        caches.keys().then(keys => {
            return Promise.all(
                keys.map(key => {
                    if (key !== CACHE_NAME) {
                        return caches.delete(key);
                    }
                })
            );
        }).then(() => self.clients.claim())
    );
});

// 拦截请求阶段：只要本地闪存里有，就绝不走流量和网络，做到100%纯离线
self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request).then(cachedResponse => {
            if (cachedResponse) {
                return cachedResponse; // 找到缓存，直接秒开返回
            }
            return fetch(event.request); // 缓存找不到（比如新外链），才走普通网络请求
        })
    );
});