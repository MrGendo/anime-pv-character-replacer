# 素材来源与使用限制（MATERIALS）

## PV 素材

| 项 | 值 |
|---|---|
| 作品 | TVアニメ「マジカル★エクスプローラー」（Magical Explorer） |
| 素材 | 第1弾PV（2026年10月3日放送開始） |
| 来源 | https://www.youtube.com/watch?v=bQuJywMCYWg（Aniplex 公式频道 アニプレックス チャンネル） |
| 官网 | https://majieku.com |
| 规格 | 1920x1080, 23.976fps, 98.1s，yt-dlp 下载（`yt-dlp -f "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]"`） |
| 使用限制 | 原视频版权归制作委员会所有。本仓库仅作个人技术研究/笔试用途，不重新分发原视频（raw_pv 不入 git），公开仓库只包含少量引用性关键帧用于说明，遵循引用原则；任何商业使用被明确排除。 |

### 片段选取依据（VLM 帧扫描 + 镜头边界检测）

全片 53 个镜头边界（HSV 直方图差分，Bhattacharyya>0.35），帧内容扫描用 Qwen3-VL-8B（SiliconFlow）。
- **片段 A = 59.309-64.606s（5.3s）**：双人魔法阵场景，披风少年 + 金发少女（宽景 + 特写），静态为主
- **片段 B = 75.325-84.710s（9.4s）**：单人动作序列，黑夹克少年空中格斗，6 个快切镜头，高速运动

选这对片段的原因：一个静态双人、一个动态单人，运动幅度/镜头数/人数都构成最大对比，跨片段复用性验证最有说服力。

## 人设图素材（替换目标角色）

**全部为 AI 生成的原创角色**，无第三方版权风险：

| 文件 | 角色 | 生成方式 |
|---|---|---|
| assets/character_sheets/chara_aoi_sheet.png | 葵（Aoi）：銀髪短发少女， teal-blue 眼， 藏青 sailor 服 + 红丝带 | cogview-3-flash（智谱，免费档），1440x720 三视图 prompt |
| assets/character_sheets/chara_jun_sheet.png | 淳（Jun）：金发少年，绿眼，黑色学兰 | 同上 |

- 生成记录：`assets/character_sheets/sheet_gen_meta.json`（模型、耗时）
- prompt 全文见 `experiments/` 下实验配置
- 使用限制：AI 生成原创角色，可自由用于本仓库演示；不主张与任何现有动画角色相似

## 外部依赖与降级

| 依赖 | 用途 | 配置 | 降级 |
|---|---|---|---|
| ffmpeg/ffprobe | 切片/合成 | PATH | 无（必需） |
| YOLO 人物检测（ONNX） | 人物 bbox | 自动下载 HF 权重，需网络 | COCO yolov8n onnx → 启发式显著性 |
| SiliconFlow API | VLM 帧扫描/盲评 | env `SILICONFLOW_API_KEY` | 跳过 VLM 评分，仅自动化指标 |
| 智谱 cogview-3-flash | 人设图生成（已完成，免费档） | env `ZHIPU_API_KEY` | 已产出入库，无需重跑 |
| 智谱 glm-4.5v | 备用 VLM | env `ZHIPU_API_KEY` | 开发期遇 429，切换 SiliconFlow |
| ARK 图像编辑（付费） | 逐帧生成式替换（未授权） | env `ARK_API_KEY` | **明确标注未运行**，仅配置位 |
