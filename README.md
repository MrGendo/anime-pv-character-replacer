# anime-pv-replacer

动画 PV 人物替换管线:PV → 切片段 → 人物定位/跟踪 → 用标准日式人设图替换人物 → 合成导出 → 评估。
当前进度:**W3(人物替换 sprite/palette 双路径 + VLM 检测 + 合成导出)**。

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
│   ├── segment.py       # CLI 入口(cut / shots 两个阶段)
│   ├── detection.py     # 人物检测(yolon ONNX / cascade 降级)
│   ├── tracking.py      # IoU 跟踪 + 丢失容忍 + bbox 平滑
│   ├── track.py         # CLI 入口(detect / track 两个阶段)
│   ├── vlm_grounding.py # VLM 人物检测(Qwen3-VL keyframe 策略,W3)
│   ├── replacement.py   # sprite / palette 双路径合成器(W3)
│   └── compose.py       # 替换合成导出 CLI(W3)
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

## VLM 人物检测(W3,可选付费路径)

yolon(COCO 照片域)对动画远景检出率仅 11%,A 段几乎不可用;W3 提供
Qwen3-VL(SiliconFlow)视觉接地检测作替代/补充:

```bash
export SILICONFLOW_API_KEY=sk-...   # 付费 API,额度自担;不设则人话提示并退出码 2
uv run python -m anime_pv_replacer.vlm_grounding --config config/pipeline.yaml --segment A
```

- 成本策略:每 `detection.grounding_interval`(默认 5)帧真实请求一次 VLM,
  中间帧按相邻两次检测线性插值 bbox;prompt 固定、temperature=0,输出解析
  失败自动重试 1 次。
- 产物与 `--stage detect` 完全同构(`frames/*.json` + `detections.json`),跑完后
  直接 `--stage track` 即可消费,不改 W2 代码。请求直连不走系统代理。

## 人物替换与合成导出(W3)

前置:片段已切出、detect/track 产物已生成。三路径二选一:

```bash
# sprite 路径:人设图白底抠像,按 bbox 缩放、脚部对齐贴入(本地 CPU,零成本)
uv run python -m anime_pv_replacer.compose --config config/pipeline.yaml --segment A --method sprite

# palette 路径:bbox 区域向人设色板(kmeans k=8)做 LAB 颜色迁移(本地 CPU,零成本)
uv run python -m anime_pv_replacer.compose --config config/pipeline.yaml --segment B --method palette \
    --chara-sheet assets/character_sheets/chara_jun_sheet.png

# api_doc 路径:未设 ARK_API_KEY 时人话提示并退出码 2;设了也只打印将要调用的
# endpoint/参数,绝不真实调用付费 API
uv run python -m anime_pv_replacer.compose --config config/pipeline.yaml --segment A --method api_doc
```

- `--track-id` 选要替换的轨迹(默认该片段最长 track);`--chara-sheet` 换人设图
  (默认取 config `replacement.chara_sheet`)。
- sprite 细节:三视图取「正面」中格、最大连通域剔除邻格越界残条;按 bbox 高度
  缩放(宽超 1.6x 钳制)、脚部对齐 bbox 底边中心、alpha 羽化贴入;宽度序列做
  滑动中值滤波(跳变 >15% 向中值回归)防闪烁;**bbox 外像素逐像素不动**。
- palette 细节:LAB 空间 Reinhard 统计量迁移(均值漂移衰减 + 对比度钳制防翻白)
  + edgePreservingFilter 去噪;区域外不动。
- 产物落 `output/{run_id}/replace/{SEG}_{method}/`:`frames/` 逐帧 jpg、
  `final.mp4`(crf/preset 取 config export 段)、`compare_viz/` 每 15 帧一张
  「原帧|替换帧」对比图、`replace_summary.json`(帧数/耗时/轨迹/人设等)。

## 测试

```bash
make test   # 等价于:uv run pytest -q
```
