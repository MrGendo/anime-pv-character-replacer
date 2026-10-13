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


## 外部 API 调用记录（成本透明）

| 时间 | 服务 | 用途 | 调用次数 | 费用 |
|---|---|---|---|---|
| 10-12 17:5x | 智谱 cogview-3-flash | 人设图生成 | 2 | 免费（flash 档） |
| 10-12 18:0x | 智谱 glm-4.5v | 帧扫描尝试 | 12 | 429 失败 0 费用 |
| 10-12 18:1x-18:4x | SiliconFlow Qwen3-VL-8B | 帧扫描 50 帧 + grounding 验证 15 帧 | ~65 | 订阅内（个人套餐，8B 模型低价档） |
| (W3) | SiliconFlow Qwen3-VL-8B | keyframe grounding（每5帧1次） | ~70 | 同上 |

付费 API（ARK doubao 图像编辑）**未授权未调用**，费用 0。
