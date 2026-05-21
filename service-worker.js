const CACHE_NAME = 'yolo-百科离线应用-v4';

const STATIC_ASSETS = [
    './',
    './index.html',
    './app.js',
    './info_db.json',
    './yolov5_n.onnx',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm.wasm',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-threaded.wasm',
    // 确保这俩文件严格区分大小写，且在仓库真实存在！
    './images/person.jpg',
    './images/bottle.jpg'
];

self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => {
            // 使用 no-cors 模式缓存外部 CDN 资源，防止跨域报错中断整个缓存过程
            return Promise.all(STATIC_ASSETS.map(url => {
                const request = new Request(url, { mode: url.startsWith('http') ? 'no-cors' : 'cors' });
                return fetch(request).then(response => {
                    if (!response.ok && response.type !== 'opaque') {
                        throw new Error(`资源请求失败: ${url}`);
                    }
                    return cache.put(request, response);
                }).catch(err => console.warn('文件缓存跳过:', url, err));
            }));
        }).then(() => self.skipWaiting())
    );
});

self.addEventListener('activate', event => {
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

self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request).then(cachedResponse => {
            if (cachedResponse) return cachedResponse; 
            return fetch(event.request); 
        })
    );
});
