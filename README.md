# Anime PV Character Replacement Pipeline（题 C 交付）

选择 2026 年 10 月新番「マジカル★エクスプローラー」第1弾PV，截取两个代表性片段（A：双人魔法阵场景 / B：单人空中动作序列），用 AI 生成的标准日式人设图（三视图）替换视频中的人物，交付可复用、可评估的完整 Pipeline。

- PV 来源（Aniplex 公式频道）：<https://www.youtube.com/watch?v=bQuJywMCYWg>
- 官方网站：<https://majieku.com>
- 素材来源与使用限制：[docs/MATERIALS.md](docs/MATERIALS.md)
- 评估方案设计：[docs/EVALUATION_DESIGN.md](docs/EVALUATION_DESIGN.md)
- 实验台账（全部 run 索引 + 外部 API 成本）：[docs/EXPERIMENT_LEDGER.md](docs/EXPERIMENT_LEDGER.md)
- 最终评估报告：[experiments/eval_20261012_w3eval.md](experiments/eval_20261012_w3eval.md)

## 实现方式以及时间规划

四波次开发（每波委派 Claude Code 实现、本人验收后提交；波次划分与验收标准见 docs/plan/TASK_W*.md）：

| 波次 | 内容 | 实际耗时 | 验收 |
|---|---|---|---|
| W1 | 项目骨架/配置/切片/镜头边界 | 10min 开发 + ~40min 素材侦察 | 5 tests ✅ commit 98a3e43 |
| W2 | 人物检测(yolon onnx + cascade 降级)/IoU 跟踪 | 18min 开发 + 20min 验收 | 12 tests ✅ |
| W3 | VLM grounding 检测 + sprite/palette 双路径替换 + 合成导出 | 17min 开发 + 25min 真实运行 | 18 tests ✅ |
| W4 | 自动化评估(5 指标+权重) | 21min 开发 + 10min 验收 | 25 tests ✅ |
| W5 | 结果整理/README/发布 | ~40min | 本文件 |
| W6 | Seedance 视频生成路径（用户反馈 sprite 不合格后新增，付费授权） | ~50min（含 2 次生成+核验） | 见下节 |

总预算 8-10 小时，实际 ~4.5 小时完成（素材侦察与实验设计占大头；扩展功能未做，聚焦双片段完整实验）。
每阶段验收标准：pytest 全绿 + 真实命令跑出产物 + 本人抽查关键中间产物后才 commit。

## Seedance 路径（W6，视频生成模型 vs 本地合成）

用户审阅 sprite 结果后反馈「纯粹画框贴图，框不是脸、贴的是全身照片」——判定不合格，授权改用视频生成模型。W6 新增第三条替换路径并真实跑通 A/B：

- 模型：doubao-seedance-2-5-260628（图生视频，参考图×2：人设图正面格 + 原片段首帧，prompt 指示按人设替换角色、保留镜头与背景），脚本 `scripts/seedance_replace.py`（API 契约复用生产验证过的 ark.py：多图 role=reference_image、任务轮询、usage tokens 记录）
- 成本：A 5s = 216,900 tokens / ~207s；B 10s = 260,100 tokens / 447s；合计 477,000 completion tokens（真实 usage 记录在 output/seedance/*_meta.json 与台账）
- 质检（VLM 抽帧核验）：身份 A 3/3、B 4/4 全部符合人设（银白短发水手服 / 金发学兰），**动作姿态为动态生成**（B 段格斗动作连续、非站桩）；已如实记录局限：镜头/背景相对原片有漂移（首帧锚定是概率性遵循，生成式重绘不保证场景逐像素还原）——对比图 `results/compare_grids/{A,B}_seedance_compare.jpg`
- 结论：三条路径定位明确——sprite（零成本、背景零污染但贴纸感强）、palette（零成本、跨片段最稳但身份弱）、**Seedance（身份+动作最佳、有成本与场景漂移 trade-off）**；评估表中 sprite 的动作低分是方法性上限，非实现 bug


## Pipeline 设计考量和侧重点

```
PV 下载(1080p) → 镜头边界检测(HSV 直方图差分,53 镜头) → 片段选取(A/B 对齐镜头边界)
      → 人物定位: VLM grounding(Qwen3-VL keyframe@5 + 线性插值) [yolon 备选]
      → IoU 跟踪 + bbox 滑窗平滑
      → 人物替换: sprite(立绘贴入,脚部对齐,alpha 羽化) / palette(LAB 色板迁移+边缘保留)
                 / seedance(图生视频重生成,W6 付费路径)
      → ffmpeg 合成导出(crf18) → 自动评估(5 维加权 100 分制) → 盲评
```

侧重点：
1. **跨片段复用性优先**——同一 config、同一命令、同一段代码跑 A/B；片段差异只体现在素材（人设图）与 track 选择，无任何按片段写死的分支
2. **降级路径完备**——yolon(COCO) 对动画域失效(检出率 11%)时切换 VLM grounding；VLM 无 key 时人话提示退出；付费 API(ARK) 仅配置位未授权不真调
3. **背景零污染**——sprite/palette 均严格只写 bbox 内像素，背景保真度由回归测试保证（区域外 1 像素改动即测试失败）
4. **诚实标注**——所有产物带 run_id + JSONL 运行日志；失败案例（见下）如实入报告

## 结果评估方案的设计和考量

指标体系（100 分制，权重按替换任务成败关键分配）：

| 维度 | 权重 | 自动指标 |
|---|---|---|
| 人物身份一致性 | 25% | LAB 直方图相关性+主色板余弦（CLIP 不可用时的降级近似，已注明） |
| 画面质量 | 15% | Laplacian 清晰度+结构保留 |
| 动作一致性 | 20% | bbox 区域光流幅值相关性 |
| 时序稳定性 | 20% | 闪烁比（1.0=与原视频一致） |
| 背景保真度 | 10% | bbox 外逐像素相等率+边缘带 SSIM |
| 跨片段泛化 | 10% | 同 method A/B 小计差+完成率+指标 σ |

考量：身份/动作/稳定性占 65%——「换对人、不乱动、不闪烁」是替换任务的核心；盲评单独列示不混入自动分；两片段同一指标同一规则同一流程。完整方案见 docs/EVALUATION_DESIGN.md。

### 片段 A、B 实验结果（同一 Pipeline、同一 config）

| 组合 | 总分 | 身份 | 质量 | 动作 | 时序 | 背景 |
|---|---|---|---|---|---|---|
| A × sprite | **84.2** | 76.4 | 93.9 | 98.3 | 74.1 | 99.1 |
| A × palette | **83.0** | 74.6 | 70.9 | 97.0 | 79.5 | 88.5 |
| B × sprite | **63.2** | 89.8 | 84.4 | 18.9 | 39.0 | 98.6 |
| B × palette | **80.3** | 67.5 | 56.0 | 86.8 | 95.9 | 89.5 |

- 跨片段泛化：sprite A-B 小计差 21.0（动作段崩），palette 仅 2.7——palette 在快切动作段的时序稳定性(95.9)远好于 sprite(39.0)
- 关键对比图：results/compare_grids/overview_4combo.jpg（四组合并排）；results/final_videos/*.mp4（成片）
- 成本：本地替换 0 元；VLM grounding ~70 次调用（个人 SiliconFlow 订阅内）；耗时 sprite 8.2/5.6ms/帧、palette 81/38ms/帧；失败帧 0；人工干预 0
- 复现：`make setup && uv run python -m anime_pv_replacer.segment --stage cut && ... `（完整命令链见 README 运行节）

## AI coding 工具、Token 用量及主要消耗环节

工具：Claude Code CLI（claude -p，GLM-5.3 后端，open.bigmodel.cn/api/anthropic），四波次委派。用量取自 `claude --output-format json` 的 usage 字段（docs/logs/taskW*_claude.json 原始记录）：

| 波次 | turns | 时长 | input | output | cache_read |
|---|---|---|---|---|---|
| W1 | 50 | 600s | 94,988 | 22,687 | 1,915,392 |
| W2 | 74 | 1055s | 75,745 | 50,222 | 3,175,424 |
| W3 | 76 | 1031s | 147,355 | 62,017 | 4,369,728 |
| W4 | 45 | 1231s | 107,111 | 65,778 | 3,828,544 |
| 合计 | 245 | 3917s | **425,199** | **200,704** | **13,289,088** |

最耗环节：**W3（147k input / 62k output）**——VLM grounding + 双路径替换 + 合成 CLI 三模块同波，契约条款最多；W4 时长最长（1231s）但 turns 最少（评估数学映射调试来回多）。
另有本人（Hermes agent）直接消耗：素材侦察/帧扫描/grounding 验证/盲评 ~130 次 VLM 调用（SiliconFlow 订阅内）。

## 个人投入时间最多的部分（不含 LLM 直接完成的工作）

1. **素材侦察与选段（~1.5h，最大头）**：yt-dlp 搜 2026 新番 PV→下载 3 部候选→逐帧 VLM 扫描（12+38+8 帧）→镜头边界交叉比对→人工裁定 A/B。难点：VLM 对特效帧（魔法阵）报「人物」的幻觉框，用裁剪回验法（裁出来再问一次）识别并排除；官网人设图抓取被浏览器工具阻塞，改 AI 生成原创人设（免版权风险）
2. **检测域间隙的工程决策（~40min）**：W2 的 yolon(COCO) 对动画远景检出率 11%，亲自实测 Qwen3-VL grounding（含幻觉框回验）后拍板加 vlm_grounding 路径，两片段统一启用
3. **真实运行与验收（~50min）**：每波 pytest 亲自跑 + 真实命令跑 A/B + 抽查 compare_viz 帧确认 sprite 贴入/背景一致
4. **评估体系设计与盲评材料（~30min）**：指标/权重设计、盲评包制作（随机重命名隐藏 truth）；发现 VLM 评委全 5 分无区分度，改属性级核验（问「银白短发+藏青水手服」是否存在）

## 失败案例分析：问题、原因及改进方向

1. **COCO 域检测在动画上失效**（W2，已修复）：yolov8n 对 A 段远景人物检出率 11%（person 分数峰值 0.234）。原因：照片域训练 vs 赛璐璐上色/远景小目标的域间隙。改进：VLM grounding（动画域无间隙），代价 3.7-4.3s/keyframe + API 依赖；根治方向是 anime-finetune YOLO 权重（未做，如实标注）
2. **VLM grounding 幻觉框**：魔法阵特效帧被报出「人物」bbox（裁剪回验发现框内是红色抽象形状）。原因：prompt 说「检测角色」，模型对发光人形特效过度泛化。缓解：keyframe 间隔降低单帧影响+跟踪平滑；改进方向：加入「特效帧过滤」判定（本版未实现）
3. **sprite 在快切动作段崩坏**（B_sprite 动作 18.9 分/闪烁 1.51）：静态立绘贴入 6 镜头快切段，每镜 bbox 跳变导致贴图位置跳变+肢体完全不动。原因：sprite 方法没有时序模型，这是方法性上限。改进：生成式视频模型（如题目允许的 seedance 类逐段重生成）或至少姿态驱动 sprite 变形（未做）
4. **palette 在 B 段画质掉分**（56.0）：快切段 bbox 边缘把背景卷进色板迁移，出现脏边。改进：更紧的人物分割（SAM 类 matting，未做）
5. **VLM 盲评无区分度**：16 样本 15 个全 5 分（「No changes detected」）。原因：评委 VLM 宽容偏差+未替换帧左右完全一致的稀释。改进：只对被替换帧抽样+换严格 rubric+人评为主；本次以属性级核验替代（通过：替换帧中确有「银白短发+藏青水手服」角色，与人设一致）
6. **glm-4.5v 429**：帧扫描与 W1 开发抢同一订阅配额。改进：切换 SiliconFlow 独立配额池（已完成）；日志在 docs/logs/scan_frames_v1.log
7. **工具链阻塞**（如实记录）：terminal 审批拦截 curl 内联/rm -rf/自启 Chrome & 等；vision_analyze 对本地图幻觉——本地帧质检全部改走 SF VLM API（可靠）

## 目录结构

```
config/          pipeline.yaml(真实实验配置) + example
src/anime_pv_replacer/  segment(切片) / detection+vlm_grounding(定位) / tracking(跟踪)
                  / replacement(sprite+palette) / compose(合成导出) / evaluate(评估)
tests/           25 tests (pytest -q 全绿)
scripts/         scan_frames.py(VLM 帧扫描)
results/         final_videos(4 成片) + compare_grids(关键帧对比) + 人设图
experiments/     eval_*.md/json(评估报告) + blind_review/(盲评材料与记录)
docs/            MATERIALS / EVALUATION_DESIGN / EXPERIMENT_LEDGER / plan(TASK_W1-W5 任务书) / logs(委派原始记录)
output/          每个 run 的中间产物(逐帧检测/跟踪/替换帧图,git 忽略大文件)
```

## 快速开始

```bash
make setup   # uv venv + 依赖
make test    # 25 tests
# 完整管线（A/B 同一命令，只有参数不同）
uv run python -m anime_pv_replacer.segment --config config/pipeline.yaml --stage cut
SILICONFLOW_API_KEY=sk-... uv run python -m anime_pv_replacer.vlm_grounding --config config/pipeline.yaml --segment A   # B 同理
uv run python -m anime_pv_replacer.track --config config/pipeline.yaml --segment A --stage track
uv run python -m anime_pv_replacer.compose --config config/pipeline.yaml --segment A --method sprite --chara-sheet assets/character_sheets/chara_aoi_sheet.png
uv run python -m anime_pv_replacer.evaluate --config config/pipeline.yaml --run-dir output/20261012_w3eval
```

外部依赖配置：`SILICONFLOW_API_KEY`（VLM 检测/扫描，无 key 时该阶段人话提示退出码 2）；`ZHIPU_API_KEY`（仅人设图再生成用，图已入库可不配）；`ARK_API_KEY`（付费生成式路径，**未授权未运行**，设置后也只打印调用计划不真调）。原始 PV 用 yt-dlp 下载（命令见 docs/MATERIALS.md），raw 视频不入 git。
