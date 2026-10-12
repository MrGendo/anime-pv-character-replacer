# TASK W2 — 人物检测 / 跟踪 / 中间产物落盘

## 环境事实（勿猜测）
- 工作目录 /Users/Admin/Downloads/project/tiC_test，git main 分支（W1 已提交）
- uv 在 /Users/Admin/.hermes/bin/uv（PATH 里没有！subprocess 用绝对路径或先 os.environ 扩 PATH）
- venv 已建好（.venv, python 3.12），deps 已含 numpy/opencv-python/pillow/httpx/pyyaml
- 系统工具 ffmpeg/ffprobe 在 /opt/homebrew/bin
- 无 GPU 全 CPU；无 Docker
- 素材已就绪：assets/segments/A.mp4 (5.3s, 1920x1080)、assets/segments/B.mp4 (9.4s)；完整 PV assets/raw_pv/magical_explorer_pv1_1080p.mp4
- config/pipeline.yaml 已填真实时间段；W1 代码在 src/anime_pv_replacer/（config.py/segmentation.py/segment.py/logging_utils.py），测试 tests/test_segmentation.py 5 个全绿——先读这些文件熟悉契约再动手

## 交付文件清单（只许创建/修改以下文件）
- src/anime_pv_replacer/detection.py
- src/anime_pv_replacer/tracking.py
- src/anime_pv_replacer/track.py（CLI：--config --segment {A,B} --stage {detect,track}）
- tests/test_detection.py
- tests/test_tracking.py
- pyproject.toml（可追加依赖：onnxruntime、imgui 无——就 onnxruntime + 下载工具用的 tqdm 不要）
- config/pipeline.yaml、config/pipeline.example.yaml（detection/tracking 段如需细化可改，别动 segments/source）
- README.md（追加 detection/tracking 用法一节）

## 契约条款
1. detection.py：
   - 类 `PersonDetector`：`__init__(method, score_thresh, device, max_persons)`；method ∈ {"yolon", "cascade"}`（cascade= OpenCV DNN 人脸级联扩展到人体近似，作为无权重降级路径）
   - yolon 路径：从 HF 下载 yolov8n.onnx（约 12MB，URL 写死 https://huggingface.co/...或 github ultralytics release 直链均可），缓存到 `output/work/models/`；下载失败时打印人话提示并自动 fallback 到 cascade（记 warning 日志）
   - `detect(frame_bgr) -> [{"bbox": [x1,y1,x2,y2], "score": float}]`，NMS 后按 score 排序，截 max_persons
   - ONNX 推理用 onnxruntime CPU；输入 letterbox 640，输出后处理（含 NMS）自己写，不引 ultralytics 包
2. tracking.py：
   - 类 `SimpleTracker`：帧间 IoU 匹配 + 丢失容忍 `max_gap_frames`（丢了记 LOST，重现且 IoU 匹配上则续 ID）
   - `update(dets, frame_idx) -> [{"track_id", "bbox", "score", "state"}]`；平滑：bbox 用 `smoothing_window` 帧滑动均值
3. track.py CLI：
   - `--stage detect`：逐帧检测，写 output/{run_id}/detection/{SEG}/frames/ 逐帧 bbox json + 汇总 detections.json（含每帧人数、平均 score、检测耗时/帧）
   - `--stage track`：读 detect 产物，跑跟踪，写 output/{run_id}/tracking/{SEG}/tracks.json（track_id → 帧区间、逐帧 bbox）+ 每隔 N=10 帧存一张画了 bbox/track_id 的可视化 jpg 到 tracks_viz/
   - JSONL 运行日志同 W1 风格（get_run_logger）
4. 性能：1080p 帧检测（yolon CPU）预算 ~150ms/帧；若超过，加 frame_stride 配置（默认 1）并在 stride>1 时对跳过帧线性插值 bbox
5. 测试（真实小视频，不用 mock 推理结果）：
   - 用 ffmpeg lavfi 生成纯色背景+移动色块视频不可行（yolon 检不出人）→ 检测测试用 A.mp4 的首帧真实图片（assets 已在），断言 detect 返回非空、bbox 在图内；cascade 路径单独测（构造纯色帧应返回空列表不崩）
   - 跟踪测试：合成 detections 序列（两个框相向移动+一个消失重现），断言 ID 连续性、LOST→RECOVER 状态、平滑后 bbox 无跳变
6. 失败提示人话：onnxruntime 缺失/权重下载失败/ffmpeg 缺失，各给一行「装什么/配什么」提示
7. 样式同 W1：类型标注、docstring、模块顶部「W2 scope」注释。禁止创建清单外文件；禁止 git commit。

## 完成标准（自验后汇报）
- `uv run pytest -q` 全绿（含 W1 的 5 个）
- 对 A、B 各跑一次 detect+track 真实命令（yolon 路径），汇报：每片段帧数、平均检测耗时/帧、检出人数分布（多少帧 0/1/2 人）、track 数与最长 track 帧数、可视化 jpg 路径
- 汇报格式：文件清单 / pytest 原文 / A/B 真实运行输出摘要 / 偏离决策

## 禁止项
- 不做替换/合成/评估（W3+）
- 不动 scripts/、docs/MATERIALS.md、assets/character_sheets/
- 不 git commit
