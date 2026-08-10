# 中文 Agent + ACT 拆解说明页

这是一个无构建步骤、无在线依赖的原生 HTML / CSS / JS 静态页面。

## 本地打开

最简单的方法是直接双击 `index.html`。视频和缩略图都使用相对路径，缺失时会显示占位说明。

也可以从 `web/` 目录启动本地静态服务器：

```bash
python3 -m http.server 8080
```

然后访问 `http://localhost:8080/`。整个 `web/` 目录也可以直接作为 GitHub Pages 的静态内容发布。

## 补充真实媒体

把媒体文件放进 `assets/` 并使用以下文件名，页面会自动从占位状态切换为真实资源：

| Demo | 视频 | 场景缩略图 |
|---|---|---|
| Bottle | `assets/bottle-demo.mp4` | `assets/bottle-scene.png` |
| Tissue | `assets/tissue-demo.mp4` | `assets/tissue-scene.png` |
| Draw | `assets/draw-demo.mp4` | `assets/draw-scene.png` |

推荐视频使用 H.264 编码的 MP4；缩略图建议使用宽高比约 2:1 的 PNG。若想换文件名，需要同步修改 `index.html` 对应 Demo 卡片上的 `data-video` 或 `data-image` 属性。
