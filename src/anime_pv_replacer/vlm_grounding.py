# W3 scope — VLM 人物检测:Qwen3-VL(SiliconFlow chat completions)对动画帧输出人物 bbox,替代/补充 yolon。
"""VLM 视觉接地检测(契约条款 0,W2 遗留问题的修复)。

背景:yolon(COCO 照片域)对动画远景检出率仅 11%,A 段几乎不可用;实测
Qwen/Qwen3-VL-8B-Instruct 能对动画帧稳定输出人物 bbox,本模块将其接入管线。

成本策略(keyframe):逐帧调 VLM 太贵,每 ``grounding_interval`` 帧(配置
detection 段,默认 5)真实请求一次,中间帧用相邻两次检测的线性插值 bbox 补齐,
复用 track.py 的插值器保证与 frame_stride 语义一致。

产物格式与 ``track.py --stage detect`` 完全兼容(frames/*.json + detections.json),
跑完后可直接用 ``--stage track`` 消费,实现「替代 yolon」而不改 W2 代码。
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import cv2

from .config import ConfigError, load_config
from .logging_utils import get_run_logger
from .segment import new_run_id
from .segmentation import MediaToolError, probe_video
from .track import _interpolate_stride_frames

logger = logging.getLogger(__name__)

# keyframe 策略默认间隔:每 5 帧一次真实 VLM 请求(config detection.grounding_interval 可覆盖)
DEFAULT_GROUNDING_INTERVAL = 5

# 契约条款 0:prompt 固定为已实测可行的版本,勿改写措辞(改动会直接影响输出格式)
VLM_PROMPT = (
    "Locate anime characters. Output ONLY JSON array "
    '[{"bbox_2d":[x1,y1,x2,y2],"desc":"<8 words>"}] '
    "in the ORIGINAL image pixel coordinates. Frame out unclear/effect regions. "
    "If none, output []."
)

# 解析前清洗:<think> 推理段与 ```json 围栏都可能出现在回复里
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

# 请求参数(除 prompt 外仅 temperature=0 为契约要求;max_tokens 防长思考截断 JSON)
VLM_TEMPERATURE = 0.0
VLM_MAX_TOKENS = 2048
REQUEST_TIMEOUT_S = 60.0


class VLMKeyMissingError(Exception):
    """VLM API key 未设置(付费路径未授权)。"""


class VLMGroundingDetector:
    """VLM 人物检测器(SiliconFlow chat completions,base64 data url 传图)。

    Parameters 里 base_url/model/api_key_env 都可注入,生产按默认值调用;
    网络错误直接上抛(fail loud),只有「输出解析失败」走契约的重试 1 次→[] 路径。
    """

    def __init__(
        self,
        api_key_env: str = "SILICONFLOW_API_KEY",
        model: str = "Qwen/Qwen3-VL-8B-Instruct",
        base_url: str = "https://api.siliconflow.cn/v1",
        confidence_note: str = "VLM 不返回置信度,score 固定 1.0,过滤交给跟踪平滑",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.confidence_note = confidence_note
        self.api_key = os.environ.get(api_key_env, "")
        if not self.api_key:
            raise VLMKeyMissingError(
                f"环境变量 {api_key_env} 未设置,VLM 人物检测不可用。"
                f"请在 shell 或 .env 中导出该 key(SiliconFlow 付费 API,额度自担),"
                f"或改用 yolon/cascade 检测路径。"
            )
        # 每次 detect_keyframes 后更新,供 CLI 写产物摘要(避免多返回值)
        self.last_stats: dict = {}

    # ---------- 单帧检测 ----------

    def detect(self, frame_bgr) -> list[dict]:
        """检测单帧,返回 [{"bbox": [x1,y1,x2,y2], "score": 1.0, "source": "vlm"}]。

        解析失败重试 1 次;仍失败则 warning 后返回 [](契约条款 0)。
        """
        for attempt in (1, 2):
            text = self._chat(frame_bgr)
            boxes = self._parse_boxes(text, frame_bgr.shape[:2])
            if boxes is not None:
                return boxes
            logger.warning("VLM 输出解析失败(第 %d 次),原始回复片段:%.200s", attempt, text)
        logger.warning("VLM 输出连续两次解析失败,本帧按空检测处理")
        return []

    def _chat(self, frame_bgr) -> str:
        """编码图片并发起一次 chat completions 请求,返回回复文本。"""
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("帧编码为 JPEG 失败(帧数据异常)")
        data_url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": VLM_PROMPT},
                ],
            }],
            "temperature": VLM_TEMPERATURE,
            "max_tokens": VLM_MAX_TOKENS,
        }
        import httpx

        # trust_env=False:不走系统代理直连,等价 urllib ProxyHandler({}) 的绕行
        with httpx.Client(trust_env=False,
                          timeout=httpx.Timeout(connect=15, read=REQUEST_TIMEOUT_S,
                                                write=30, pool=15)) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
        if resp.status_code != 200:
            raise RuntimeError(f"VLM 请求失败 HTTP {resp.status_code}:{resp.text[:300]}")
        content = resp.json()["choices"][0]["message"]["content"]
        if isinstance(content, list):  # 防御:部分网关把 content 拆成多段
            content = "".join(part.get("text", "") for part in content)
        return str(content)

    @staticmethod
    def _parse_boxes(text: str, frame_shape) -> list[dict] | None:
        """解析回复为检测列表;结构不符返回 None(交由上层重试)。"""
        cleaned = _THINK_RE.sub("", text).strip()
        m = _FENCE_RE.search(cleaned)
        if m:
            cleaned = m.group(1).strip()
        # 防御性截取首个 '[' 到最后一个 ']',容忍模型在 JSON 外多说一句话
        lo, hi = cleaned.find("["), cleaned.rfind("]")
        if lo < 0 or hi <= lo:
            return None
        try:
            items = json.loads(cleaned[lo:hi + 1])
        except json.JSONDecodeError:
            return None
        if not isinstance(items, list):
            return None
        h, w = frame_shape[:2]
        out: list[dict] = []
        for item in items:
            box = item.get("bbox_2d") if isinstance(item, dict) else None
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in box)
            except (TypeError, ValueError):
                continue
            x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
            y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:  # 夹紧后退化成线的框丢弃
                continue
            out.append({"bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "score": 1.0, "source": "vlm"})
        return out

    # ---------- keyframe 策略 ----------

    def detect_keyframes(self, video_path: str | Path,
                         interval: int = DEFAULT_GROUNDING_INTERVAL) -> list[list[dict]]:
        """整段视频检测:每 interval 帧真实请求一次,中间帧线性插值。

        返回按帧号对齐的检测列表(与 detect 阶段的逐帧语义一致);
        运行统计写入 self.last_stats(键帧数/请求耗时等)。
        """
        interval = max(1, int(interval))
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise MediaToolError(f"OpenCV 打不开视频(编码不受支持或文件损坏):{video_path}")

        det_by_idx: dict[int, list[dict]] = {}
        vlm_ms: list[float] = []
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % interval == 0:  # 0 % interval == 0:首帧必为真实检测
                t0 = time.perf_counter()
                det_by_idx[frame_idx] = self.detect(frame)
                vlm_ms.append((time.perf_counter() - t0) * 1000.0)
            frame_idx += 1
        cap.release()
        if frame_idx == 0:
            raise MediaToolError(f"视频一帧都没读到(文件损坏?):{video_path}")

        per_frame = _interpolate_stride_frames(det_by_idx, sorted(det_by_idx), frame_idx)
        self.last_stats = {
            "n_frames": frame_idx,
            "n_keyframes": len(det_by_idx),
            "grounding_interval": interval,
            "avg_vlm_ms": round(sum(vlm_ms) / len(vlm_ms), 2) if vlm_ms else 0.0,
            "max_vlm_ms": round(max(vlm_ms), 2) if vlm_ms else 0.0,
        }
        return [per_frame[i] for i in range(frame_idx)]


def run_ground_stage(cfg: dict, run_id: str, out_root: Path, seg: str) -> None:
    """对片段跑 VLM keyframe 检测,落 detect 阶段兼容产物(frames/ + detections.json)。"""
    logger = get_run_logger(run_id, out_root / "logs", stage=f"vlm-{seg}")

    video = Path(cfg["paths"]["assets_root"]) / "segments" / f"{seg}.mp4"
    if not video.is_file():
        raise FileNotFoundError(f"片段视频不存在:{video.resolve()}\n请先运行 segment --stage cut 生成片段。")

    info = probe_video(video)
    interval = max(1, int(cfg["detection"].get("grounding_interval", DEFAULT_GROUNDING_INTERVAL)))
    detector = VLMGroundingDetector()
    logger.event("vlm.start", segment=seg, video=str(video), model=detector.model,
                 grounding_interval=interval, n_frames=info["n_frames"])

    per_frame = detector.detect_keyframes(video, interval)

    frames_dir = out_root / "detection" / seg / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    hist: dict[str, int] = {}
    for idx, dets in enumerate(per_frame):
        with open(frames_dir / f"{idx:06d}.json", "w", encoding="utf-8") as f:
            json.dump({"frame_idx": idx, "detections": dets,
                       "interpolated": idx % interval != 0}, f, ensure_ascii=False, indent=2)
        hist[str(len(dets))] = hist.get(str(len(dets)), 0) + 1

    stats = detector.last_stats
    summary = {
        "segment": seg,
        "source": str(video),
        "run_id": run_id,
        "method": "vlm",
        "model": detector.model,
        "confidence_note": detector.confidence_note,
        "grounding_interval": interval,
        "fps": info["fps"],
        "width": info["width"],
        "height": info["height"],
        "n_frames": stats["n_frames"],
        "n_keyframes": stats["n_keyframes"],
        "avg_vlm_ms": stats["avg_vlm_ms"],
        "max_vlm_ms": stats["max_vlm_ms"],
        "person_count_hist": hist,
    }
    out_json = out_root / "detection" / seg / "detections.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.event("vlm.done", segment=seg, n_frames=stats["n_frames"],
                 n_keyframes=stats["n_keyframes"], avg_vlm_ms=stats["avg_vlm_ms"],
                 person_count_hist=hist, detections_json=str(out_json))


def main(argv: list[str] | None = None) -> int:
    """CLI 入口,返回退出码(0 成功 / 1 失败 / 2 VLM key 未授权)。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="anime_pv_replacer.vlm_grounding",
        description="动画 PV 人物替换管线 — VLM 人物检测(keyframe 降成本,W3)",
    )
    parser.add_argument("--config", required=True, help="配置文件路径(YAML)")
    parser.add_argument("--segment", required=True, choices=["A", "B"],
                        help="目标片段名(对应 config.segments 里的 name)")
    args = parser.parse_args(argv)

    run_id = new_run_id()
    try:
        cfg = load_config(args.config)
        out_root = Path(cfg["paths"]["output_root"]) / run_id
        run_ground_stage(cfg, run_id, out_root, args.segment)
    except VLMKeyMissingError as e:
        # 契约条款 0:VLM 无 key 只给人话提示,退出码固定 2
        print(f"[失败] {e}", file=sys.stderr)
        return 2
    except (ConfigError, FileNotFoundError, MediaToolError, RuntimeError, ValueError) as e:
        print(f"[失败] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
