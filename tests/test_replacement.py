# W3 scope — replacement/compose 契约测试:sprite 抠像、bbox 外背景保真、palette 几何保持、compose 冒烟。
"""W3 契约测试。

sprite/palette 用真实人设图 + 合成帧(不依赖视频,保证 bbox 外像素
逐像素可断言);compose 冒烟用 A.mp4 前 3 帧 + cascade 检测(不要求
yolon 权重,避免测试依赖网络),跑通「合成 3 帧小视频」全流程。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from anime_pv_replacer.compose import _start_encoder, main as compose_main
from anime_pv_replacer.detection import PersonDetector
from anime_pv_replacer.replacement import (
    PaletteReplacer,
    SpriteReplacer,
    extract_sprite,
    stabilize_widths,
)
from anime_pv_replacer.segmentation import probe_video

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHEET_AOI = PROJECT_ROOT / "assets/character_sheets/chara_aoi_sheet.png"
SEG_A = PROJECT_ROOT / "assets/segments/A.mp4"


def _synthetic_frame(h: int = 270, w: int = 480) -> np.ndarray:
    """确定性合成帧:双正弦渐变 + 竖条纹,有结构、无随机性,便于逐像素断言。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    frame = np.zeros((h, w, 3), np.uint8)
    frame[..., 0] = (140 + 60 * np.sin(xx / 37.0)).astype(np.uint8)   # B 横向渐变
    frame[..., 1] = (140 + 60 * np.cos(yy / 53.0)).astype(np.uint8)   # G 纵向渐变
    frame[..., 2] = ((xx // 24) % 2 * 90 + 80).astype(np.uint8)       # R 竖条纹
    return frame


def _ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """标准 SSIM(灰度,11x11 均匀窗口,0-255 尺度)—— 仅测试用,W4 评估另做。"""
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    win = (11, 11)
    mu_a, mu_b = cv2.blur(a, win), cv2.blur(b, win)
    var_a = cv2.blur(a * a, win) - mu_a ** 2
    var_b = cv2.blur(b * b, win) - mu_b ** 2
    cov = cv2.blur(a * b, win) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float((num / den).mean())


def test_extract_sprite() -> None:
    """真实人设图抠像:RGBA 四通道、四角透明、meta 记录中格切分区域。"""
    sprite, meta = extract_sprite(SHEET_AOI)
    assert sprite.ndim == 3 and sprite.shape[2] == 4, "sprite 应为 RGBA"
    h, w = sprite.shape[:2]
    corners = [sprite[0, 0, 3], sprite[0, w - 1, 3], sprite[h - 1, 0, 3], sprite[h - 1, w - 1, 3]]
    assert all(c == 0 for c in corners), f"四角像素应透明,实际 alpha={corners}"
    assert (sprite[..., 3] > 0).any(), "sprite 应有可见内容"
    # 1440 宽三等分 → 正面格为 [480, 960)
    assert meta["cell_x_range"] == [480, 960], f"正面应取中格,实际 {meta['cell_x_range']}"
    assert h > w, "全身立绘应高大于宽"


def test_sprite_bbox_unchanged_outside() -> None:
    """背景保真回归:bbox 外像素与原图逐像素相等(契约条款 1 硬性边界)。"""
    frame = _synthetic_frame()
    bbox = [150.0, 30.0, 290.0, 250.0]
    out = SpriteReplacer(SHEET_AOI, edge_blend_px=8, feather_sigma=3.0).replace(frame, bbox)
    assert out.shape == frame.shape

    x1, y1 = int(round(bbox[0])), int(round(bbox[1]))
    x2, y2 = int(round(bbox[2])), int(round(bbox[3]))
    # bbox 内外都复制后抹掉内侧,比较其余全部像素
    a, b = out.copy(), frame.copy()
    a[y1:y2, x1:x2] = 0
    b[y1:y2, x1:x2] = 0
    assert np.array_equal(a, b), "bbox 外像素必须与原图逐像素相等"
    assert not np.array_equal(out[y1:y2, x1:x2], frame[y1:y2, x1:x2]), "bbox 内应有替换内容"


def test_sprite_rejects_empty_bbox() -> None:
    """空/退化 bbox:原样返回且不抛异常(契约条款 1)。"""
    frame = _synthetic_frame()
    replacer = SpriteReplacer(SHEET_AOI)
    for bad in ([], None, [10.0, 10.0, 10.0, 10.0], [0, 0, -5, 9]):  # type: ignore[list-item]
        out = replacer.replace(frame, bad)
        assert out is frame, f"非法 bbox {bad} 应原样返回同一帧"


def test_palette_keeps_geometry() -> None:
    """palette 路径:bbox 区域 SSIM>0.5(结构未大变)且区域外逐像素相等。"""
    frame = _synthetic_frame()
    bbox = [150.0, 40.0, 330.0, 250.0]
    out = PaletteReplacer(SHEET_AOI).replace(frame, bbox)

    x1, y1 = int(round(bbox[0])), int(round(bbox[1]))
    x2, y2 = int(round(bbox[2])), int(round(bbox[3]))
    a, b = out.copy(), frame.copy()
    a[y1:y2, x1:x2] = 0
    b[y1:y2, x1:x2] = 0
    assert np.array_equal(a, b), "palette 只允许动 bbox 内像素"

    ssim = _ssim_gray(frame[y1:y2, x1:x2], out[y1:y2, x1:x2])
    assert ssim > 0.5, f"bbox 区域结构应基本保持,SSIM={ssim:.3f}"
    assert not np.array_equal(out[y1:y2, x1:x2], frame[y1:y2, x1:x2]), "颜色应发生迁移"


def test_stabilize_widths() -> None:
    """宽度滑动中值滤波:单帧尖峰被拉回中值,缓变趋势不受影响(契约条款 3)。"""
    spiked = [100.0, 100.0, 100.0, 100.0, 160.0, 100.0, 100.0]
    out = stabilize_widths(spiked)
    assert out[4] == 100.0, f"尖峰帧应回归窗口中值,实际 {out[4]}"
    assert all(w == 100.0 for i, w in enumerate(out) if i != 4)

    ramp = [100.0, 110.0, 120.0, 130.0, 140.0]  # 每帧 ~8-9% 渐变,不属闪烁
    assert stabilize_widths(ramp) == ramp, "缓变趋势不应被改写"


def test_compose_smoke(tmp_path: Path) -> None:
    """A.mp4 前 3 帧 + cascade 检测(或固定 bbox 兜底)跑通 sprite/palette 合成小视频。

    不要求 yolon 权重(避免网络依赖);产物结构与 replace_summary.json 契约齐全。
    """
    cap = cv2.VideoCapture(str(SEG_A))
    frames = []
    while len(frames) < 3:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    assert len(frames) == 3, "A.mp4 应至少能读出 3 帧"

    # cascade 检测第一帧;检不出时用固定 bbox 兜底(cascade 对远景动画弱,属预期)
    h, w = frames[0].shape[:2]
    dets = PersonDetector(method="cascade", score_thresh=0.1).detect(frames[0])
    bbox = [round(v, 1) for v in dets[0]["bbox"]] if dets \
        else [round(0.32 * w, 1), round(0.15 * h, 1), round(0.62 * w, 1), round(0.92 * h, 1)]

    # 3 帧迷你 A.mp4(ffmpeg rawvideo 编码,不碰 assets/)
    mini_assets = tmp_path / "assets"
    mini_assets.joinpath("segments").mkdir(parents=True)
    mini_video = mini_assets / "segments" / "A.mp4"
    enc = _start_encoder(mini_video, w, h, 23.976, 23, "veryfast")
    for frame in frames:
        enc.stdin.write(frame.tobytes())
    enc.stdin.close()
    assert enc.wait() == 0

    # 假 tracks.json:track 1 覆盖全部 3 帧(bbox 固定)
    out_root = tmp_path / "output"
    track_dir = out_root / "run0" / "tracking" / "A"
    track_dir.mkdir(parents=True)
    track_payload = {
        "segment": "A", "n_frames": 3, "n_tracks": 1,
        "tracks": [{
            "track_id": 1, "frame_start": 0, "frame_end": 2,
            "frames": [{"frame_idx": i, "bbox": bbox, "score": 0.9, "state": "TRACKING"}
                       for i in range(3)],
        }],
    }
    (track_dir / "tracks.json").write_text(json.dumps(track_payload), encoding="utf-8")

    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(
        f"""
version: "0.1"
paths:
  assets_root: {mini_assets}
  output_root: {out_root}
replacement:
  method: sprite
  chara_sheet: {SHEET_AOI}
  edge_blend_px: 6
  feather_sigma: 3
export:
  fps: 23.976
  crf: 23
  preset: veryfast
""",
        encoding="utf-8",
    )

    for method in ("sprite", "palette"):
        assert compose_main(["--config", str(cfg_path), "--segment", "A",
                             "--method", method]) == 0, f"{method} 合成应成功"

        run_dir = sorted(out_root.glob("*/replace/A_" + method))[-1]
        final = run_dir / "final.mp4"
        assert final.is_file() and final.stat().st_size > 0, "final.mp4 应非空"
        assert probe_video(final)["n_frames"] == 3, "小视频应恰 3 帧"
        assert len(list((run_dir / "frames").glob("*.jpg"))) == 3, "逐帧图应 3 张"
        assert len(list((run_dir / "compare_viz").glob("*.jpg"))) == 1, "3 帧只抽第 0 帧做对比图"

        summary = json.loads((run_dir / "replace_summary.json").read_text(encoding="utf-8"))
        assert summary["method"] == method and summary["track_id"] == 1
        assert summary["n_frames"] == 3 and summary["n_replaced_frames"] == 3
        assert summary["avg_replace_ms"] >= 0 and summary["total_s"] > 0
        assert "chara_sheet" in summary and "final_mp4" in summary
