# anime-pv-replacer

动画 PV 人物替换管线:PV → 切片段 → 人物定位/跟踪 → 用标准日式人设图替换人物 → 合成导出 → 评估。
当前进度:**W1(项目骨架 / 配置系统 / 切片与镜头边界检测)**。

## 目录结构

```
├── assets/
│   ├── raw_pv/          # 原始 PV 素材
│   ├── segments/        # 切出的片段(cut 阶段产物)
│   └── character_sheets/# 标准人设图(后续波次使用)
├── config/
│   ├── pipeline.yaml        # 主配置(实际使用)
│   └── pipeline.example.yaml# 示例配置(注释丰富)
├── docs/
│   └── plan/            # 任务规划文档
├── output/              # 每次 run 的产物:{run_id}/{stage}/ 与日志
├── scripts/             # 辅助脚本
├── src/anime_pv_replacer/
│   ├── config.py        # 配置加载(YAML + env 定位 + deep_merge)
│   ├── logging_utils.py # 结构化 JSONL 运行日志
│   ├── segmentation.py  # ffprobe 探测 / ffmpeg 切片 / 镜头边界检测
│   └── segment.py       # CLI 入口(cut / shots 两个阶段)
└── tests/               # pytest 测试
```

## 安装

需要 Python 3.12+、[uv](https://docs.astral.sh/uv/)、ffmpeg/ffprobe(macOS:`brew install ffmpeg`)。

```bash
make setup        # 等价于:uv venv --python 3.12 && uv sync
```

另外复制 `.env.example` 为 `.env` 并填写 `ARK_API_KEY`(替换阶段的风格化 API 密钥,W1 阶段尚不需要)。

## 一条命令跑切片

```bash
make segment
# 或显式指定:
uv run python -m anime_pv_replacer.segment --config config/pipeline.yaml --stage cut
```

按 `config/pipeline.yaml` 的 `segments` 列表把 PV 切成 `assets/segments/*.mp4`,
每个片段的时长会经 ffprobe 复核,误差超过 0.2s 记 warning。

镜头边界检测:

```bash
uv run python -m anime_pv_replacer.segment --config config/pipeline.yaml --stage shots
```

产物写入 `output/{run_id}/detection/shots.json`,运行日志(JSONL)在 `output/{run_id}/logs/`。

## 人物检测与跟踪(W2)

先确保片段已切出(`--stage cut`),再按「先 detect 后 track」两步跑:

```bash
# 逐帧人物检测(yolon 路径,首次运行自动下载 yolov8n.onnx 约 12MB 到 output/work/models/)
uv run python -m anime_pv_replacer.track --config config/pipeline.yaml --segment A --stage detect

# IoU 跟踪(自动读取最新一次 detect 产物)
uv run python -m anime_pv_replacer.track --config config/pipeline.yaml --segment A --stage track
```

- detect 产物:`output/{run_id}/detection/{SEG}/frames/`(逐帧 bbox json)+
  `detections.json`(每帧人数、平均 score、检测耗时/帧、人数分布直方图)。
- track 产物:`output/{run_id}/tracking/{SEG}/tracks.json`(每条轨迹的帧区间与逐帧
  平滑 bbox、状态 TRACKING/LOST/RECOVER)+ `tracks_viz/` 每 10 帧一张画框可视化 jpg。
- 检测双路径:`yolon`(onnxruntime CPU 推理,手写 letterbox/NMS 后处理,实测 1080p
  约 17ms/帧)与 `cascade`(OpenCV Haar 人脸级联外扩人体近似,无权重降级路径)。
  yolon 权重下载失败或库缺失时自动降级 cascade 并记 warning。
- 阈值说明:`score_thresh` 默认 0.15 —— yolov8n 是 COCO 照片域模型,对动画画面
  置信度整体偏低(远景段峰值仅 ~0.23),0.35 会把远景片段全部滤空。
- CPU 预算超支(>150ms/帧)时可调 `detection.frame_stride > 1`:每 N 帧真实推理一次,
  跳过帧按相邻两次检测线性插值(IoU 配对后过渡)。

## 测试

```bash
make test   # 等价于:uv run pytest -q
```
