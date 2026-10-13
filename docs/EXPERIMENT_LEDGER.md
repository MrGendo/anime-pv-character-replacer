# 实验台账（EXPERIMENT LEDGER）

所有 run 的索引。评分/报告以本文件 + experiments/ 下 JSON 为准，手工不编辑自动产物。

| run_id | 日期 | 内容 | 产物路径 | 状态 |
|---|---|---|---|---|
| 20261012_180852 | 10-12 | W1 全片镜头边界（53 镜头） | output/20261012_180852/detection/shots.json | ✅ |
| 20261012_182613 | 10-12 | A/B 按真实镜头边界切出 | assets/segments/{A,B}.mp4 | ✅ |
| 20261012_184353/184403 | 10-12 | W2 yolon 检测 A/B（A 检出率 11%，域间隙） | output/*/detection/{A,B}/detections.json | ⚠️ A 段不可用 |
| 20261012_184421 | 10-12 | W2 IoU 跟踪 A/B | output/20261012_184421/tracking/{A,B}/tracks.json | ⚠️ 同上 |
| (W3 run) | 10-12 | vlm grounding + sprite/palette 替换 | output/{run}/replace/… | 进行中 |
| 20261013_0938 | 10-13 | Seedance A 段生成（5s/720p） | output/seedance/A_seedance.mp4 + _meta.json | ✅ 身份符合/场景部分漂移 |
| 20261013_0948 | 10-13 | Seedance B 段生成（10s/720p） | output/seedance/B_seedance.mp4 + _meta.json | ✅ 身份符合/动态保留/场景漂移 |

## Seedance 路径（10-13 新增，用户授权付费调用）

用户反馈 sprite 贴图效果不合格（「框的位置不是人脸、贴的是全身照片」），授权改用视频生成模型（doubao-seedance-2-5-260628，图生视频：人设图+原片段首帧双参考图）。脚本 scripts/seedance_replace.py。

| 段 | 时长 | tokens | 墙钟 | 成片 |
|---|---|---|---|---|
| A | 5s | 216,900 | ~207s | results/final_videos/A_seedance.mp4（6.3MB） |
| B | 10s | 260,100 | 447s | results/final_videos/B_seedance.mp4（13.2MB） |

合计 477,000 completion tokens。VLM 抽帧核验：身份（发色/瞳色/服装）A 3/3、B 4/4 符合人设；动作姿态为动态生成（非站桩贴图）；已知局限：镜头/背景与原片有漂移（生成式重绘不保证场景还原，prompt 锚定首帧背景仍是概率性遵循）。

## 10-13 第二轮：动作保真迭代（首尾帧锚定 → reference_video 视频输入）

用户反馈首轮生成「基本没有动作」，要求以 A 段原 5s 为基础、动作基本不变、人物换成 Jun。文档站被反爬拦截，改用 **API 报错探针**逆向出 Seedance 2.5 的隐藏能力与规则：

| 探针 | 报错 | 逆向出的规则 |
|---|---|---|
| content 传 data:video base64 | `reference media mode requires video role to be reference_video` | 存在 r2v 模式，role=reference_video |
| video_url 用 base64 | `reference_video must be provided as a web url` | 视频必须公网 URL |
| 640x360 探针视频 | `video pixel count must be >= 407696` | 输入视频像素数下限 |

| 实验 | 模式 | tokens | 结果 |
|---|---|---|---|
| A_jun_shot1 | 首尾帧锚定（子镜头1，2s 请求被钳到 5s） | 108,900 | 首帧 corr=1.000（锚定生效）；动作幅度 0.38×（拉伸变慢） |
| A_jun_full | 首尾帧锚定整段 5s | 324,900 | 成片 11.3MB（用户中断，未细评） |
| **A_jun_refvideo** | **reference_video（整段原片直喂）** | **216,900** | **动作幅度 0.97×（逐帧光流贴合原片）；镜头1 帧 corr 0.95-0.99；镜头2 特写重构（corr~0.4，构图漂移）；金发质心在画面中上（Jun 已应用）** |

结论：**r2v（reference_video）是「动作基本不变」的正解**——源视频直接作为条件输入，光流轨迹逐帧对齐（0.97）；残余问题只剩特写镜头的构图漂移。源视频经 GitHub raw 分支（probe-assets）做公网 URL。脚本：scripts/seedance_framed.py（首尾帧）、scripts/seedance_refvideo.py（r2v，推荐）。
今日累计 1,127,700 tokens（5 次生成）。


## 外部 API 调用记录（成本透明）

| 时间 | 服务 | 用途 | 调用次数 | 费用 |
|---|---|---|---|---|
| 10-12 17:5x | 智谱 cogview-3-flash | 人设图生成 | 2 | 免费（flash 档） |
| 10-12 18:0x | 智谱 glm-4.5v | 帧扫描尝试 | 12 | 429 失败 0 费用 |
| 10-12 18:1x-18:4x | SiliconFlow Qwen3-VL-8B | 帧扫描 50 帧 + grounding 验证 15 帧 | ~65 | 订阅内（个人套餐，8B 模型低价档） |
| (W3) | SiliconFlow Qwen3-VL-8B | keyframe grounding（每5帧1次） | ~70 | 同上 |

付费 API（ARK doubao 图像编辑）**未授权未调用**，费用 0。
