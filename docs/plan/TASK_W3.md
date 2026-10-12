# TASK W3 — 人物替换（sprite / palette 双路径）与合成导出

## 环境事实（勿猜测）
- 工作目录 /Users/Admin/Downloads/project/tiC_test，git main（W1/W2 已提交）
- uv=/Users/Admin/.hermes/bin/uv；ffmpeg/ffprobe=/opt/homebrew/bin；venv=.venv(py3.12, 含 numpy/opencv/httpx/pyyaml/onnxruntime)
- 无 GPU 全 CPU
- W2 产物结构（先读 src/anime_pv_replacer/detection.py、tracking.py、track.py 熟悉接口）：
  - output/{run_id}/detection/{SEG}/detections.json
  - output/{run_id}/tracking/{SEG}/tracks.json（track_id → frames[{frame_idx, bbox, score, state}]）
- 人设图：assets/character_sheets/chara_aoi_sheet.png、chara_jun_sheet.png（1440x720 三视图，白底，AI 生成原创角色）
- 素材：assets/segments/A.mp4（双人：披风少年+金发少女）、assets/segments/B.mp4（单人：黑夹克少年）

## 交付文件清单
- src/anime_pv_replacer/replacement.py（sprite 与 palette 两类合成器）
- src/anime_pv_replacer/compose.py（CLI：--config --segment {A,B} --method {sprite,palette} [--track-id N] [--chara-sheet path]）
- tests/test_replacement.py
- README.md（追加替换/合成用法）
- config/pipeline.yaml / example（replacement 段更新为三路径：sprite/palette/api_doc，api_doc 保持「未授权未运行」标注）

## 契约条款
1. replacement.py：
   - `extract_sprite(sheet_path, view="front") -> (sprite_rgba, meta)`：白底转 alpha（阈值+羽化），三视图取正面那一格（按 1/3 宽度切分，meta 记录切分区域）
   - 类 `SpriteReplacer`：`__init__(chara_sheet, edge_blend_px, feather_sigma)`；`replace(frame_bgr, bbox) -> frame_bgr`：sprite 按 bbox 高度缩放（宽按比例，若宽超 bbox 1.6x 则钳制），脚部对齐 bbox 底边中心，alpha 羽化贴入；**bbox 外像素严格不动**（背景保真度依赖此条）
   - 类 `PaletteReplacer`：`__init__(chara_sheet)`；提取人设图主色板（kmeans k=8）；`replace(frame_bgr, bbox)`：bbox 区域内向色板做 LAB 空间颜色迁移 + cv2.edgePreservingFilter，区域外不动
   - 两个类都拒绝空 bbox（返回原帧+warning 日志）
2. compose.py CLI 流程：读 W2 tracks.json（--track-id 指定要替换的轨迹，默认该片段最长 track）→ 逐帧（ffmpeg 解码）调用 replacer → 逐帧替换图存 output/{run_id}/replace/{SEG}_{method}/frames/ → ffmpeg 编码 final.mp4（crf/preset 从 config export 段）→ side-by-side 对比图（原帧|替换帧）每 15 帧一张存 compare_viz/ → replace_summary.json（帧数、方法、track_id、人设、总耗时、逐帧耗时均值）
3. 时序：bbox 用 W2 已平滑的；sprite 缩放宽度相邻帧跳变>15% 时向均值回归（防闪烁），实现为滑动中值滤波
4. api_doc 路径：compose.py --method api_doc 时若 ARK_API_KEY 未设置→人话提示「付费路径未授权，见 docs/MATERIALS.md」并退出码 2；设置了也只打印将要调用的 endpoint/参数并退出（不许真调付费 API）
5. 测试：
   - test_extract_sprite：真实人设图→alpha 通道存在、四角像素透明
   - test_sprite_bbox_unchanged_outside：合成一帧后，bbox 外像素与原图逐像素相等（背景保真的回归测试）
   - test_palette_keeps_geometry：palette 路径输出的 bbox 区域与原图 SSIM>0.5（结构未大变）且区域外相等
   - test_compose_smoke：对 A.mp4 用 cascade 检测+固定 bbox 合成 3 帧小视频跑通（不要求 yolon 权重，避免测试依赖网络）
6. 样式同 W1/W2；模块顶「W3 scope」。禁止清单外文件；禁止 git commit。

## 完成标准（自验后汇报）
- `uv run pytest -q` 全绿
- 真实跑 4 个组合：A×sprite、A×palette、B×sprite、B×palette（各一条命令+输出摘要：帧数、逐帧耗时、final.mp4 路径与大小、compare_viz 路径）
- 汇报格式：文件清单 / pytest 原文 / 4 组合运行输出摘要 / 偏离决策

## 禁止项
- 不做评估（W4）
- 不动 scripts/、docs/MATERIALS.md、assets/
- 不 git commit
