# W2 scope — 人物检测:yolon(ONNX YOLOv8n CPU 推理,自写 letterbox/NMS 后处理)与 cascade(OpenCV 人脸级联扩展到人体近似,无权重降级路径)。
"""人物检测模块。

yolon 路径不引 ultralytics 包:ONNX 推理用 onnxruntime,letterbox 预处理、
cxcywh→xyxy 解码、按 person 类(COCO id=0)过滤、NMS 全部手写,
保证依赖面最小且行为可单测。权重(yolov8n.onnx,约 12MB)按序尝试
MODEL_URLS 各直链下载,缓存到 ``{workdir_root}/models/``;所有失败路径
(库缺失/下载失败/文件损坏)统一降级 cascade 并记 warning,不中断管线。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 权重下载直链(HF 官方域名优先,hf-mirror 为直连不通时的镜像兜底;两个域名指向同一文件)
MODEL_URLS = [
    "https://huggingface.co/kshitijjjjjjjjjjjjjjjj/yolov8n-coco-onnx/resolve/main/yolov8n.onnx",
    "https://hf-mirror.com/kshitijjjjjjjjjjjjjjjj/yolov8n-coco-onnx/resolve/main/yolov8n.onnx",
]
MODEL_FILE = "yolov8n.onnx"
MODEL_MIN_BYTES = 1_000_000  # 权重合法性下限(~12MB 的模型不可能小于 1MB),防 CDN 错误页被当权重缓存

LETTERBOX_SIZE = 640   # YOLOv8 导出模型的固定输入边长
LETTERBOX_PAD = 114    # ultralytics 约定的 letterbox 填充灰度
PERSON_CLASS_ID = 0    # COCO 类别 0 = person(本管线只关心人)
NMS_IOU = 0.45         # NMS 的 IoU 阈值(同一目标重复框抑制)

# cascade 路径的人脸→人体近似比例:站姿人体约 7.5 个头高,肩/身宽约 2.6 个头宽
# (放宽取值以盖住披风/裙摆等动画服装外扩;该路径本就是降级近似,不追求精确)
BODY_HEIGHT_HEADS = 7.5
BODY_WIDTH_HEADS = 2.6


class DetectionError(Exception):
    """检测器初始化或推理失败(两条路径都不可用)。"""


class _YolonUnavailable(Exception):
    """yolon 路径暂不可用(库缺失/下载失败/权重损坏),触发降级 cascade。"""


def _letterbox(frame: np.ndarray, size: int = LETTERBOX_SIZE) -> tuple[np.ndarray, float, int, int]:
    """等比缩放并居中填充到 size x size,返回 (画布, 缩放比, x 填充, y 填充)。

    返回的三个几何量供检测后把 640 空间坐标映回原图:x_orig = (x_640 - pad_x) / scale。
    """
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), LETTERBOX_PAD, dtype=np.uint8)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def _download_model(urls: list[str], dst: Path) -> None:
    """按序尝试直链下载权重到 dst(先写 .part 再原子改名,避免半截文件被当缓存)。"""
    import httpx

    dst.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for url in urls:
        try:
            with httpx.Client(follow_redirects=True,
                              timeout=httpx.Timeout(connect=15, read=300, write=30, pool=15)) as client:
                with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    tmp = dst.with_suffix(".part")
                    done = 0
                    with open(tmp, "wb") as f:
                        for chunk in resp.iter_bytes(1 << 16):
                            f.write(chunk)
                            done += len(chunk)
            if done < MODEL_MIN_BYTES:
                raise RuntimeError(f"下载内容异常(仅 {done} 字节,疑似错误页)")
            tmp.replace(dst)
            logger.info("权重下载完成:%s(%d 字节)", dst, done)
            return
        except Exception as e:  # 网络类异常一律试下一个源,全部失败才上抛
            last_err = e
            logger.warning("权重下载失败:%s(%s)", url, e)
    raise _YolonUnavailable(
        f"所有下载源均失败:{last_err}\n"
        f"可手动下载 yolov8n.onnx 放到 {dst} 后重试,本次先降级 cascade 路径。"
    )


class PersonDetector:
    """人物检测器:yolon(默认)或 cascade 降级路径。

    Parameters 里的 models_dir / model_urls 为带默认值的扩展入参:
    生产调用只按契约传前 4 个;测试用它们把缓存与下载源指向临时位置。
    """

    def __init__(
        self,
        method: str = "yolon",
        score_thresh: float = 0.35,
        device: str = "cpu",
        max_persons: int = 4,
        *,
        models_dir: str | Path | None = None,
        model_urls: list[str] | None = None,
    ) -> None:
        if method not in {"yolon", "cascade"}:
            raise ValueError(f"未知的检测方法:{method}(支持 yolon / cascade)")
        self.score_thresh = float(score_thresh)
        self.device = device
        self.max_persons = int(max_persons)
        # 实际生效的方法:yolon 初始化失败会改写为 cascade,调用方应读此字段而非回读入参
        self.method = method

        if method == "yolon":
            try:
                self._init_yolon(models_dir, model_urls)
            except _YolonUnavailable as e:
                # 契约:下载失败打印人话提示并自动 fallback 到 cascade(记 warning 日志)
                logger.warning("yolon 不可用,降级 cascade:%s", e)
                print(f"[警告] yolon 路径不可用,已降级 cascade:{e}", flush=True)
                self.method = "cascade"
        if self.method == "cascade":
            self._init_cascade()

    # ---------- 初始化 ----------

    def _init_yolon(self, models_dir: str | Path | None, model_urls: list[str] | None) -> None:
        """准备 onnxruntime 会话与权重;任何失败抛 _YolonUnavailable 由上层降级。"""
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise _YolonUnavailable("缺少 onnxruntime,请运行 uv sync 安装项目依赖后重试。") from e

        # 权重缓存目录默认走配置的 workdir(output/work/models),测试可覆盖
        dst = Path(models_dir) if models_dir else Path("output/work/models")
        model_path = dst / MODEL_FILE
        if not model_path.is_file():
            _download_model(model_urls or MODEL_URLS, model_path)

        # 按入参 device 选 provider;请求的不可用(如本机无 GPU)则回退 CPU 并记 warning
        wanted = {"cpu": ["CPUExecutionProvider"],
                  "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                  "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"]}.get(
                      self.device.lower(), ["CPUExecutionProvider"])
        providers = [p for p in wanted if p in ort.get_available_providers()]
        if not providers:
            logger.warning("设备 %s 无可用 provider,回退 CPU", self.device)
            providers = ["CPUExecutionProvider"]

        try:
            self._session = ort.InferenceSession(str(model_path), providers=providers)
        except Exception as e:
            # 缓存文件损坏:删掉让下次运行重新下载,本次降级
            model_path.unlink(missing_ok=True)
            raise _YolonUnavailable(f"权重加载失败(已删除缓存待重下):{e}") from e
        self._input_name = self._session.get_inputs()[0].name

    def _init_cascade(self) -> None:
        """级联路径:opencv-python 自带的 Haar 人脸级联(无网络依赖的兜底)。"""
        xml = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self._cascade = cv2.CascadeClassifier(xml)
        if self._cascade.empty():  # opencv-python 发行版必带该文件,仅防御性检查
            raise DetectionError(f"Haar 级联文件缺失或损坏:{xml}")

    # ---------- 推理 ----------

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """检测单帧,返回 [{"bbox": [x1,y1,x2,y2], "score": float}]。

        bbox 为原图像素坐标(float);NMS 在各路径内部完成后,这里统一做
        score 阈值过滤、按 score 降序排序、截断到 max_persons。
        """
        if self.method == "cascade":
            dets = self._detect_cascade(frame_bgr)
        else:
            dets = self._detect_yolon(frame_bgr)
        dets = [d for d in dets if d["score"] >= self.score_thresh]
        dets.sort(key=lambda d: d["score"], reverse=True)
        return dets[: self.max_persons]

    def _detect_yolon(self, frame_bgr: np.ndarray) -> list[dict]:
        """ONNX 推理 + 手写后处理(letterbox 逆映射 / person 过滤 / NMS)。"""
        h, w = frame_bgr.shape[:2]
        canvas, scale, pad_x, pad_y = _letterbox(frame_bgr)
        # BGR→RGB、HWC→CHW、/255,与 ultralytics 导出时的预处理保持一致
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        preds = self._session.run(None, {self._input_name: blob})[0]

        preds = preds[0] if preds.ndim == 3 else preds      # 去掉 batch 维 → [84, 8400]
        if preds.shape[0] == 84:                            # 84 在前则转置成「每行一框」
            preds = preds.T
        boxes, scores = preds[:, :4], preds[:, 4 + PERSON_CLASS_ID]
        keep_mask = scores >= self.score_thresh
        boxes, scores = boxes[keep_mask], scores[keep_mask]
        if boxes.shape[0] == 0:
            return []

        # cxcywh(letterbox 空间)→ xyxy,再逆 letterbox 映回原图并夹紧到画面内
        cx, cy, bw, bh = boxes.T
        x1 = np.clip((cx - bw / 2 - pad_x) / scale, 0, w - 1)
        y1 = np.clip((cy - bh / 2 - pad_y) / scale, 0, h - 1)
        x2 = np.clip((cx + bw / 2 - pad_x) / scale, 0, w - 1)
        y2 = np.clip((cy + bh / 2 - pad_y) / scale, 0, h - 1)

        # NMS 只在 person 分数上进行,等价于先按类过滤再做类内 NMS
        keep = cv2.dnn.NMSBoxes(
            bboxes=np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist(),
            scores=scores.tolist(),
            score_threshold=self.score_thresh,
            nms_threshold=NMS_IOU,
        )
        idxs = np.array(keep).reshape(-1) if len(keep) else []
        return [
            {"bbox": [round(float(x1[i]), 1), round(float(y1[i]), 1),
                      round(float(x2[i]), 1), round(float(y2[i]), 1)],
             "score": round(float(scores[i]), 4)}
            for i in idxs
        ]

    def _detect_cascade(self, frame_bgr: np.ndarray) -> list[dict]:
        """人脸级联 + 按头身比例外扩成人体近似框(无权重降级路径,精度弱于 yolon)。

        detectMultiScale3 的 level_weights 是级联置信度(经验范围约 1~5),
        归一到 0~1 后作为 score,使其与 yolon 路径的阈值/排序语义一致。
        """
        gray = cv2.equalizeHist(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
        min_face = max(16, frame_bgr.shape[1] // 40)  # 人脸最小边随分辨率缩放,1080p 约 48px
        rects, _, weights = self._cascade.detectMultiScale3(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(min_face, min_face), outputRejectLevels=True,
        )
        results: list[dict] = []
        for (fx, fy, fw, fh), weight in zip(rects, weights):
            bw = fw * BODY_WIDTH_HEADS
            x1, y1 = fx + fw / 2 - bw / 2, float(fy)       # 发顶视为人体顶
            x2, y2 = fx + fw / 2 + bw / 2, fy + fh * BODY_HEIGHT_HEADS
            h, w = frame_bgr.shape[:2]
            results.append({
                "bbox": [round(max(0.0, x1), 1), round(max(0.0, y1), 1),
                         round(min(float(w), x2), 1), round(min(float(h), y2), 1)],
                "score": round(min(1.0, float(weight) / 5.0), 4),
            })
        return results
