# W4 scope — evaluation/evaluate 契约测试:五维指标区分度、背景 1 像素回归、CLI 最小 fixture。
"""W4 契约测试。

指标测试全部用确定性合成图(相同/完全不同/轻微噪声三档,契约条款 3),
不依赖真实视频与 VLM;evaluate CLI 用 2 帧最小 run-dir fixture 跑通
JSON+MD 产出,同时覆盖「光流帧对不足 → 维度 N/A 不中断」路径。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from anime_pv_replacer.compose import _start_encoder
from anime_pv_replacer.evaluation import (
    background_fidelity,
    clip_identity_score,
    flicker_score,
    motion_consistency,
    piecewise_score,
    quality_score,
    temporal_flicker,
)
from anime_pv_replacer.evaluate import main as evaluate_main

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHEET_AOI = PROJECT_ROOT / "assets/character_sheets/chara_aoi_sheet.png"


def _synthetic_frame(h: int = 96, w: int = 128, phase: float = 0.0) -> np.ndarray:
    """确定性合成帧:细密竖纹 + 双正弦渐变,保证 Laplacian 方差非零。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    frame = np.zeros((h, w, 3), np.uint8)
    frame[..., 0] = (140 + 60 * np.sin((xx + phase) / 9.0)).astype(np.uint8)
    frame[..., 1] = (140 + 60 * np.cos((yy + phase) / 23.0)).astype(np.uint8)
    frame[..., 2] = ((xx // 6) % 2 * 90 + 80).astype(np.uint8)
    return frame


def _write_frames(directory: Path, frames: list[np.ndarray]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(directory / f"{i:06d}.png"), frame)
    return directory


# ---------- 指标区分度(契约条款 3:相同→高分,不同→低分) ----------

def test_clip_identity_discriminates(tmp_path: Path) -> None:
    """身份降级近似:替换区与人设同配色 → 高分;互补色 → 低分;非法 bbox → None。"""
    sheet = tmp_path / "sheet.png"
    img = np.full((200, 300, 3), 255, np.uint8)
    img[30:170, 110:190] = (40, 60, 200)  # 前景必须落在三视图的中格 [100,200)
    cv2.imwrite(str(sheet), img)

    region = np.full((96, 128, 3), 40, np.uint8)
    region[20:80, 20:110] = (40, 60, 200)   # 与人设前景同色
    diff = region.copy()
    diff[20:80, 20:110] = (60, 200, 60)     # 互补色

    same = clip_identity_score(region, region, sheet, [20, 20, 110, 80])
    other = clip_identity_score(region, diff, sheet, [20, 20, 110, 80])
    assert same is not None and same > 0.9, f"同配色应高分,实际 {same}"
    assert other is not None and other < 0.2, f"互补色应低分,实际 {other}"
    assert clip_identity_score(region, region, sheet, [10, 10, 10, 10]) is None, "退化 bbox 应返回 None"


def test_quality_score_discriminates() -> None:
    """画质:相同→高分,轻微噪声→居中,模糊/结构完全错位→显著更低;非法帧→None。"""
    sharp = _synthetic_frame()
    same = quality_score(sharp, sharp)
    blurred = quality_score(cv2.GaussianBlur(sharp, (15, 15), 4.0), sharp)
    rng = np.random.default_rng(7)
    noisy = np.clip(sharp.astype(np.int16) + rng.integers(-25, 26, sharp.shape), 0, 255).astype(np.uint8)
    noised = quality_score(noisy, sharp)
    shifted = quality_score(np.roll(sharp, (23, 41), axis=(0, 1)), sharp)
    assert same is not None and same > 0.85, f"相同帧应高分,实际 {same}"
    assert blurred is not None and blurred < same - 0.2, f"模糊应显著更低,实际 {blurred}"
    assert noised is not None and same - 0.05 > noised > 0.5, f"轻微噪声应居中,实际 {noised}"
    assert shifted is not None and shifted < same - 0.2, f"结构错位应显著更低,实际 {shifted}"
    assert quality_score(None) is None, "非法帧应返回 None"


def test_motion_consistency(tmp_path: Path) -> None:
    """动作一致性:运动序列一致 → 1.0;一方静止 → 0;双静止 → 1.0;目录缺失 → None。

    bbox 固定不动(若 bbox 紧跟目标会把 crop 内运动抵消掉),方块以变步幅
    7/3/10 px 穿过区域,保证幅值序列有方差。
    """
    steps = (7, 3, 10)

    def moving() -> list[np.ndarray]:
        frames, x = [], 10
        for i in range(4):
            f = _synthetic_frame()
            f[30:50, x:x + 20] = (240, 240, 240)
            frames.append(f)
            if i < 3:
                x += steps[i]
        return frames

    def frozen() -> list[np.ndarray]:
        frames = []
        for _ in range(4):
            f = _synthetic_frame()
            f[30:50, 10:30] = (240, 240, 240)
            frames.append(f)
        return frames

    boxes = {i: [8.0, 20.0, 60.0, 70.0] for i in range(4)}
    orig = _write_frames(tmp_path / "orig", moving())
    same = _write_frames(tmp_path / "same", moving())
    static_dir = _write_frames(tmp_path / "frozen", frozen())

    v = motion_consistency(orig, same, boxes)
    assert v is not None and v > 0.99, f"运动一致应 ≈1,实际 {v}"
    v = motion_consistency(orig, static_dir, boxes)
    assert v is not None and v < 0.05, f"一方静止应 ≈0,实际 {v}"
    v = motion_consistency(_write_frames(tmp_path / "os", [_synthetic_frame()] * 4), static_dir, boxes)
    assert v == 1.0, "双方都静止应视为完全一致"
    assert motion_consistency(tmp_path / "nope", same, boxes) is None, "目录缺失应返回 None"


def test_temporal_flicker(tmp_path: Path) -> None:
    """闪烁比:完全一致 → 1.0;替换序列抖动更强 → 比值>1;原静止替换闪 → 封顶。"""
    base = _synthetic_frame()
    boxes = {i: [20, 20, 100, 80] for i in range(4)}

    def seq(delta: int) -> list[np.ndarray]:
        return [base if i % 2 == 0 else np.clip(base.astype(np.int16) + delta, 0, 255).astype(np.uint8)
                for i in range(4)]

    orig = _write_frames(tmp_path / "fo", seq(4))
    same = _write_frames(tmp_path / "fs", seq(4))
    wild = _write_frames(tmp_path / "fw", seq(20))

    assert temporal_flicker(orig, same, boxes) == 1.0, "完全一致应为 1.0"
    ratio = temporal_flicker(orig, wild, boxes)
    assert ratio is not None and 4.5 < ratio < 5.5, f"±20 对 ±4 的抖动比应 ≈5,实际 {ratio}"
    static_dir = _write_frames(tmp_path / "fst", [base] * 4)
    capped = temporal_flicker(static_dir, wild, boxes)
    assert capped is not None and capped > 10, "原视频静止而替换闪烁应封顶重罚"


def test_background_fidelity_one_pixel(tmp_path: Path) -> None:
    """背景保真回归(契约条款 3):区域外改 1 像素必须 <1.0;完全相同 = 1.0。"""
    frame = _synthetic_frame()
    bbox = [20, 20, 100, 80]
    assert background_fidelity(frame, frame.copy(), bbox) == 1.0, "相同帧应为 1.0"
    assert background_fidelity(frame, frame.copy()) == 1.0, "无 bbox 全图模式相同帧也应为 1.0"

    touched = frame.copy()
    touched[5, 10, 0] = np.uint8(int(touched[5, 10, 0]) ^ 0xFF)  # bbox 外 (5,10) 改 1 像素
    v = background_fidelity(frame, touched, bbox)
    assert v is not None and v < 1.0, f"区域外 1 像素改动必须 <1.0,实际 {v}"

    assert background_fidelity(frame, frame[:, :64], bbox) is None, "尺寸不一致应返回 None"


def test_score_mapping() -> None:
    """分段线性映射节点 + 闪烁反向映射(契约条款 2)。"""
    assert piecewise_score(0.0) == 0.0
    assert piecewise_score(0.5) == 60.0
    assert piecewise_score(0.75) == 80.0
    assert piecewise_score(1.0) == 100.0
    assert piecewise_score(0.625) == 70.0
    assert piecewise_score(-1.0) == 0.0 and piecewise_score(2.0) == 100.0, "输入钳制到 [0,1]"
    assert flicker_score(1.0) == 100.0, "闪烁比 1.0 = 与原视频一致 = 满分"
    assert flicker_score(2.0) == 0.0
    assert flicker_score(0.5) == 40.0


# ---------- evaluate CLI 最小 fixture(契约条款 3:2 帧跑通 JSON+MD) ----------

def test_evaluate_cli_minimal_fixture(tmp_path: Path) -> None:
    """2 帧最小 run-dir:出 eval JSON+MD;motion 因仅 1 个帧对记 N/A 不中断;
    盲评 CSV 并入 md 单独一节,坏行跳过。"""
    h, w = 96, 128
    frames = [_synthetic_frame(phase=0.0), _synthetic_frame(phase=3.0)]
    bbox = [32.0, 24.0, 96.0, 72.0]

    assets = tmp_path / "assets"
    assets.joinpath("segments").mkdir(parents=True)
    enc = _start_encoder(assets / "segments" / "A.mp4", w, h, 23.976, 23, "veryfast")
    for frame in frames:
        enc.stdin.write(frame.tobytes())
    enc.stdin.close()
    assert enc.wait() == 0

    out_root = tmp_path / "output"
    tracks_path = out_root / "w3" / "tracking" / "A" / "tracks.json"
    tracks_path.parent.mkdir(parents=True)
    tracks_path.write_text(json.dumps({
        "segment": "A", "n_tracks": 1,
        "tracks": [{"track_id": 1, "frame_start": 0, "frame_end": 1,
                    "frames": [{"frame_idx": i, "bbox": bbox, "score": 0.9, "state": "TRACKING"}
                               for i in range(2)]}],
    }), encoding="utf-8")

    combo = out_root / "evalrun" / "replace" / "A_sprite"
    combo.joinpath("frames").mkdir(parents=True)
    for i, frame in enumerate(frames):
        replaced = frame.copy()
        cv2.rectangle(replaced, (32, 24), (95, 71), (200, 80, 120), -1)
        cv2.imwrite(str(combo / "frames" / f"{i:06d}.jpg"), replaced,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    combo.joinpath("replace_summary.json").write_text(json.dumps({
        "segment": "A", "method": "sprite", "run_id": "evalrun", "track_id": 1,
        "track_frames": [0, 1], "chara_sheet": str(SHEET_AOI),
        "tracks_source": str(tracks_path), "n_frames": 2, "n_replaced_frames": 2,
        "avg_replace_ms": 1.2, "max_replace_ms": 2.0, "total_s": 0.5,
        "width_stabilized_pulled": 0,
    }), encoding="utf-8")

    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(f"""
version: "0.1"
paths:
  assets_root: {assets}
  output_root: {out_root}
""", encoding="utf-8")

    blind_csv = tmp_path / "blind.csv"
    blind_csv.write_text(
        "segment,method,dim,score,comment\n"
        "A,sprite,identity,4,发型配色对得上\n"
        "A,sprite,quality,3,边缘略硬\n"
        "A,sprite,background,abc,坏行应被跳过\n",
        encoding="utf-8")

    rc = evaluate_main(["--config", str(cfg_path), "--run-dir", str(out_root / "evalrun"),
                        "--blind", str(blind_csv), "--out-dir", str(tmp_path / "experiments")])
    assert rc == 0, "最小 fixture 评估应成功"

    payload = json.loads((tmp_path / "experiments" / "eval_evalrun.json").read_text(encoding="utf-8"))
    md = (tmp_path / "experiments" / "eval_evalrun.md").read_text(encoding="utf-8")
    combo_out = payload["combos"][0]
    assert combo_out["metrics"]["motion"] is None, "2 帧只有 1 个光流对,应记 N/A"
    assert combo_out["metrics"]["background"] is not None
    assert combo_out["scores"]["identity"] is not None
    assert combo_out["total"] is not None and 0.0 <= combo_out["total"] <= 100.0
    assert len(payload["blind"]) == 2, "坏行被跳过,应只剩 2 行盲评"
    assert combo_out["cost"]["completion"] == 1.0
    for text in ("人物身份一致性", "A_sprite", "N/A", "盲评", "成本记录", "发型配色对得上"):
        assert text in md, f"md 报告应包含:{text}"
