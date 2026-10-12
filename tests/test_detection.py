# W2 scope — detection 契约测试:cascade 纯色帧空检、yolon 真实首帧检出、权重下载失败自动降级。
"""PersonDetector 契约测试。

不 mock 推理结果:cascade 路径喂真实纯色帧;yolon 路径读 assets/segments/A.mp4
首帧真实图片跑完整 ONNX 推理(权重缓存到项目 output/work/models/,与 CLI 共享,
只下载一次)。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from anime_pv_replacer.detection import PersonDetector

SEG_A = Path("assets/segments/A.mp4")
needs_segment_a = pytest.mark.skipif(
    not SEG_A.is_file(), reason="需要 W1 cut 产物 assets/segments/A.mp4"
)


def make_detector(method: str, score_thresh: float = 0.35, **kwargs) -> PersonDetector:
    """按契约签名构造检测器(4 个契约入参 + 测试扩展入参)。"""
    return PersonDetector(method=method, score_thresh=score_thresh, device="cpu",
                          max_persons=4, **kwargs)


def test_cascade_pure_color_returns_empty() -> None:
    """纯色帧无任何人脸特征,cascade 路径应返回空列表且不崩。"""
    det = make_detector("cascade")
    assert det.method == "cascade"
    frame = np.full((720, 1280, 3), 127, dtype=np.uint8)
    assert det.detect(frame) == []


@needs_segment_a
def test_yolon_real_first_frame() -> None:
    """A.mp4 首帧(双人远景)真实推理:检出非空、bbox 在图内、score 降序且达标。

    阈值用 0.15 而非管线默认的 0.35:yolov8n 是 COCO 照片域模型,对动画远景
    置信度整体偏低(A 段全片 person 分数峰值仅 0.234),首帧实测峰值约 0.22。
    """
    det = make_detector("yolon", score_thresh=0.15)
    assert det.method == "yolon"  # 真实环境(onnxruntime + 可下载权重)不应降级
    cap = cv2.VideoCapture(str(SEG_A))
    ok, frame = cap.read()
    cap.release()
    assert ok, "A.mp4 首帧读取失败"

    dets = det.detect(frame)
    assert len(dets) >= 1, f"首帧应检出至少 1 人,实际 {dets}"
    h, w = frame.shape[:2]
    scores = [d["score"] for d in dets]
    assert scores == sorted(scores, reverse=True), "结果应按 score 降序"
    assert len(dets) <= 4, "结果应截断到 max_persons"
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h, f"bbox 越界:{d['bbox']}"
        assert d["score"] >= 0.15, f"score 低于阈值:{d['score']}"


def test_yolon_download_fail_falls_back_to_cascade(tmp_path: Path) -> None:
    """所有下载源失败(指向必然拒绝连接的本地端口)时:记 warning 并降级 cascade。"""
    det = make_detector("yolon",
                        models_dir=tmp_path / "models",
                        model_urls=["http://127.0.0.1:9/none.onnx"])
    assert det.method == "cascade", "下载失败应自动降级 cascade"
    # 降级后检测器仍可用:纯色帧返回空列表不崩
    assert det.detect(np.full((480, 640, 3), 50, dtype=np.uint8)) == []
