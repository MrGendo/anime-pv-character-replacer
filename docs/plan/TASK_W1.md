# TASK W1 — 项目骨架 / 配置系统 / 切片与镜头边界检测

## 环境事实（勿猜测，按此实现）
- 工作目录：/Users/Admin/Downloads/project/tiC_test（已 git init，main 分支，空仓库）
- Python：用 `uv` 管理。`uv venv --python 3.12`（本地 uv 已装 cpython 3.12.13）。运行用 `uv run python ...`
- 系统工具：ffmpeg 9 / ffprobe 已在 PATH。无 Docker。
- 网络代理：shell 需 `export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897` 才能访问外网；Python 脚本内直连 API 用 ProxyHandler({}) 绕代理（本机 localhost 不走代理）。
- 无 GPU。全部 CPU 可跑。
- 已有素材：assets/raw_pv/magical_explorer_pv1_1080p.mp4（1920x1080, 23.976fps, 98.1s，マジカル★エクスプローラー第1弾PV）。

## 项目背景（一句话）
动画 PV 人物替换管线：PV → 切片段 → 人物定位/跟踪 → 用标准日式人设图替换人物 → 合成导出 → 评估。W1 只做地基：配置+切片+镜头边界。

## 交付文件清单（只许创建/修改以下文件）
- pyproject.toml（uv 项目，包名 anime_pv_replacer，deps: numpy, opencv-python, pillow, httpx; dev-deps: pytest）
- src/anime_pv_replacer/__init__.py（版本号 0.1.0）
- src/anime_pv_replacer/config.py — 配置加载
- src/anime_pv_replacer/segmentation.py — 切片+镜头边界
- src/anime_pv_replacer/logging_utils.py — 结构化运行日志
- src/anime_pv_replacer/segment.py — CLI 入口（`python -m anime_pv_replacer.segment`）
- tests/test_segmentation.py
- config/pipeline.yaml — 主配置模板
- config/pipeline.example.yaml — 示例配置（注释丰富）
- .env.example
- README.md（骨架版：项目一句话、目录结构、安装、一条命令跑切片）
- Makefile（targets: setup, test, segment）

## 契约条款（逐条核对实现）
1. config.py：
   - `load_config(path)` -> dataclass 或 dict（选一种，写明）。支持 YAML；未提供 path 时找 env `ANIME_PV_CONFIG`，再退 ./config/pipeline.yaml。
   - 提供 `deep_merge(base, override)`：嵌套 dict 合并，override 覆盖 base，None 值不覆盖。
   - 必含字段（pipeline.yaml 定义）：version、source（video_path, video_url, license_note）、segments（list: name, start, end, note）、paths（assets_root, output_root, workdir_root）、detection（method: "yolon", score_thresh, device: "cpu", max_persons: 4）、replacement（method: "stylization", api: {...endpoint,key_env, model, retries, timeout_s}, edge_blend_px: 12, feather_sigma: 8）、tracking（smoothing_window: 7, max_gap_frames: 8）、export（fps, crf: 18, preset: "slow"）、evaluation（metrics: [...], weights）。
   - key 一律从 env 读（key_env: "ARK_API_KEY"），代码里出现明文 key 即违规。
2. segmentation.py：
   - `probe_video(path)` -> {width, height, fps, n_frames, duration_s, codec}（用 ffprobe json 输出解析，不 subprocess 拼字符串猜）。
   - `cut_segment(src, start, end, dst)`：ffmpeg -ss before -i -t，stream copy 优先，失败自动降级重编码；返回 dst 与实际 out_time（ffprobe 验证时长误差 < 0.2s，超差记 warning 日志）。
   - `detect_shot_boundaries(src, threshold=0.35, min_gap_s=1.0)`：帧差分直方图法（HSV hist, HISTCMP_BHATTACHARYYA），返回 [(shot_start_s, shot_end_s), ...] 并写中间产物 shots.json。
3. segment.py CLI：
   - `uv run python -m anime_pv_replacer.segment --config config/pipeline.yaml --stage cut` — 按 config segments 切出 assets/segments/*.mp4。
   - `--stage shots` — 跑镜头边界检测并把 shots.json 写到 output/{run_id}/detection/shots.json。
   - run_id 格式 `YYYYmmdd_HHMMSS`；每个 run 输出目录 output/{run_id}/{stage}/。
   - 每步写 docs/logs/（或 output/{run_id}/logs/）下 JSONL 运行日志：{ts, run_id, stage, event, data}。
4. logging_utils.py：`get_run_logger(run_id, outdir)` 返回带 `.event(name, **data)` 方法的 logger，写 JSONL + 同步 print 到 stdout。
5. 失败提示：所有 API/外部依赖（ffmpeg、yolon 权重下载、ARK endpoint）失败时给一句人话提示（哪个依赖、怎么配、env 名），不带 traceback 轰炸。
6. 测试（pytest，真实文件小用例，不用 mock 代替一切）：
   - test_config_deep_merge / test_config_load_defaults
   - test_probe_video（用 ffmpeg 生成 2s testsrc 视频后 probe）
   - test_cut_segment_duration（切 0-1s，误差<0.2s）
   - test_shot_boundaries（用 ffmpeg 拼接两个不同 testsrc 生成 4s 视频，检测出 >=1 个边界）
   - 测试里生成的临时媒体放 tmp_path，不留垃圾。
7. 样式：类型标注、docstring、每个模块顶部注释「W1 scope」。禁止创建清单外文件；禁止 git commit。

## 完成标准（自验后才汇报）
- `uv run pytest -q` 全绿（贴真实输出）。
- `uv run python -m anime_pv_replacer.segment --config config/pipeline.yaml --stage cut` 真实跑通：按 segments 切出 2 个 mp4（config 里预填 A=15-20s, B=60-65s 占位，注释「时间以镜头扫描结果为准」）。
- 汇报格式：文件清单 / pytest 输出原文 / 真实运行输出原文 / 偏离决策。

## 禁止项
- 不实现人物检测/替换/评估（后续波次）。
- 不动 assets/、docs/plan/、scripts/scan_frames.py。
- 不 git commit；不创建 .venv 以外的大文件。
