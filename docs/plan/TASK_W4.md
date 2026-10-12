# TASK W4 — 自动化评估 + 报告产物生成

## 环境事实（勿猜测）
- 工作目录 /Users/Admin/Downloads/project/tiC_test，git main（W1-W3 已提交）
- uv=/Users/Admin/.hermes/bin/uv；venv 已含 numpy/opencv/httpx/pyyaml/onnxruntime
- W3 产物：output/{run_id}/replace/{SEG}_{method}/ 下 frames/（逐帧替换图）、final.mp4、compare_viz/（原|替换对比图）、replace_summary.json
- W2 产物：output/{run_id}/tracking/{SEG}/tracks.json
- 原片段：assets/segments/A.mp4、B.mp4
- 人设图：assets/character_sheets/chara_aoi_sheet.png、chara_jun_sheet.png
- 评估设计文档：docs/EVALUATION_DESIGN.md（先读，指标/权重以此为准）
- VLM 可用后端：SiliconFlow Qwen3-VL-8B-Instruct（env SILICONFLOW_API_KEY；chat completions, image base64 data url）——仅用于可选的 VLM 辅助身份评分，没 key 就跳过
- 先读 src/anime_pv_replacer/ 现有模块（config/detection/tracking/replacement/compose/logging_utils）再动手

## 交付文件清单
- src/anime_pv_replacer/evaluation.py
- src/anime_pv_replacer/evaluate.py（CLI：--config --run-dir output/{run_id} [--blind 「5分制人工盲评CSV路径」]）
- tests/test_evaluation.py
- README.md（追加评估用法）
- experiments/ 下生成 A/B×sprite/palette 的评估汇总（运行时产物，不手工编）

## 契约条款
1. evaluation.py 指标（全部 numpy/opencv 实现，不引 torch）：
   - `clip_identity_score(orig_frame, replaced_frame, sheet_path, bbox)`：**不调 CLIP**（无 GPU 不装 torch），替代实现：人设图与替换后人像区在 LAB 空间做直方图相关性（cv2.HISTCMP_CORREL）+ 主色板余弦相似度，输出 0-1；docstring 明确「CLIP 不可用时的降级近似」
   - `quality_score(frame_bgr)`：Laplacian 方差（清晰度）+ 替换前后差（结构保留），0-1
   - `motion_consistency(orig_frames_dir, replaced_frames_dir, bbox_per_frame)`：替换前后 bbox 区域光流幅值（cv2.calcOpticalFlowFarneback 相邻帧）相关性，0-1
   - `temporal_flicker(orig_frames_dir, replaced_frames_dir, bbox_per_frame)`：替换区相邻帧均值差的序列与原视频同序列的比值，1.0=与原视频一致，>1 闪烁
   - `background_fidelity(orig, replaced)`：bbox 外逐像素相等率（应=1.0）+ 边缘带 SSIM
   - 全部指标：输入不合法返回 None 并记 warning，不抛异常中断整轮
2. evaluate.py CLI：
   - 输入 run-dir，扫 replace/{SEG}_{method}/，逐组合算指标，输出 experiments/eval_{run_id}.json + eval_{run_id}.md（人读表格：指标原始值/映射分/加权总分，两片段并排）
   - 分段线性映射：0.0→0分,0.5→60分,0.75→80分,1.0→100分（flicker 用 |1-x| 反向）；权重按 EVALUATION_DESIGN.md
   - --blind CSV（columns: segment,method,dim,score,comment）存在时并入 md 表（盲评单独一节，不混自动分）
   - 跨片段泛化：同 method 的 A/B 总分差与关键指标方差，写入表尾
   - 成本记录：从 replace_summary.json 汇总每帧耗时/失败帧/人工干预次数
3. 测试：
   - 合成小图对（相同/完全不同/轻微噪声）验证每个指标的区分度（相同→高分，不同→低分）
   - background_fidelity 对「区域外改了 1 像素」的用例必须 <1.0（回归测试）
   - evaluate.py 对一个最小 run-dir fixture（2 帧）跑通出 JSON+MD
4. 样式同前；模块顶「W4 scope」。禁止清单外文件；禁止 git commit。

## 完成标准（自验后汇报）
- `uv run pytest -q` 全绿
- 真实跑 evaluate.py 对 W3 的 4 组合（A/B × sprite/palette）出表，汇报总分与各维度分（原文表格）
- 汇报格式：文件清单 / pytest 原文 / 评估表原文 / 偏离决策

## 禁止项
- 不做 GitHub 发布/README 总报告（W5）
- 不动 docs/EVALUATION_DESIGN.md、docs/MATERIALS.md、scripts/
- 不 git commit
