# W1 scope — 契约条款 6 要求的 5 个测试:全部用 ffmpeg 真实生成的媒体,不用 mock。
"""config / segmentation 契约测试。

测试媒体一律用 ffmpeg lavfi(testsrc/testsrc2)现场生成、放 pytest tmp_path,
跑完即回收,不留垃圾。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from anime_pv_replacer.config import deep_merge, load_config
from anime_pv_replacer.segmentation import (
    cut_segment,
    detect_shot_boundaries,
    probe_video,
)

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(
    FFMPEG is None, reason="测试需要 ffmpeg(本机已按任务约定预装)"
)


def make_testsrc(path: Path, pattern: str = "testsrc", seconds: float = 2.0) -> Path:
    """用 lavfi 源生成小尺寸测试视频(320x240@10fps,libx264)。"""
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"{pattern}=size=320x240:rate=10:duration={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True, text=True,
    )
    return path


# ---------- config ----------

def test_config_deep_merge() -> None:
    """嵌套 dict 递归合并;None 不覆盖;list 整体覆盖。"""
    base = {"a": 1, "nested": {"x": 1, "y": 2}, "keep": "base", "items": [1, 2]}
    override = {"a": 9, "nested": {"y": 99, "z": None}, "items": [3]}
    merged = deep_merge(base, override)
    assert merged["a"] == 9                       # 标量覆盖
    assert merged["nested"] == {"x": 1, "y": 99}  # dict 递归合并
    assert "z" not in merged["nested"]            # None 值不落地
    assert merged["items"] == [3]                 # list 整体覆盖
    assert merged["keep"] == "base"               # 未提及的键保留
    assert base["nested"]["y"] == 2               # base 不被就地修改


def test_config_load_defaults(tmp_path: Path) -> None:
    """用户配置只写少量键时,其余字段回退默认值。"""
    cfg_file = tmp_path / "mini.yaml"
    cfg_file.write_text(
        yaml.safe_dump({"version": "9.9", "segments": [{"name": "X", "start": 1, "end": 2}]}),
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg["version"] == "9.9"                          # 用户值生效
    assert [s["name"] for s in cfg["segments"]] == ["X"]    # list 整体来自用户
    assert cfg["detection"]["method"] == "yolon"            # 未写 → 默认
    assert cfg["detection"]["max_persons"] == 4
    assert cfg["replacement"]["api"]["key_env"] == "ARK_API_KEY"
    assert cfg["replacement"]["edge_blend_px"] == 12
    assert cfg["tracking"]["smoothing_window"] == 7
    assert cfg["export"]["crf"] == 18
    assert cfg["export"]["preset"] == "slow"


# ---------- segmentation ----------

@needs_ffmpeg
def test_probe_video(tmp_path: Path) -> None:
    """生成 2s testsrc 后 probe,核验元信息。"""
    video = make_testsrc(tmp_path / "src.mp4", seconds=2.0)
    info = probe_video(video)
    assert info["width"] == 320
    assert info["height"] == 240
    assert abs(info["fps"] - 10.0) < 0.1
    assert abs(info["duration_s"] - 2.0) < 0.2
    assert 18 <= info["n_frames"] <= 22
    assert info["codec"] == "h264"


@needs_ffmpeg
def test_cut_segment_duration(tmp_path: Path) -> None:
    """切 0-1s,产物时长误差 < 0.2s。"""
    src = make_testsrc(tmp_path / "src.mp4", seconds=2.0)
    dst, actual = cut_segment(src, 0.0, 1.0, tmp_path / "out" / "seg.mp4")
    assert dst.is_file()
    assert abs(actual - 1.0) < 0.2
    assert abs(probe_video(dst)["duration_s"] - 1.0) < 0.2


@needs_ffmpeg
def test_shot_boundaries(tmp_path: Path) -> None:
    """两种 testsrc 各 2s 拼成 4s 视频,应检出 >= 1 个边界(即 >= 2 个 shot)。"""
    part_a = make_testsrc(tmp_path / "a.mp4", pattern="testsrc", seconds=2.0)
    part_b = make_testsrc(tmp_path / "b.mp4", pattern="testsrc2", seconds=2.0)
    concat_list = tmp_path / "list.txt"
    concat_list.write_text(
        f"file '{part_a}'\nfile '{part_b}'\n", encoding="utf-8"
    )
    joined = tmp_path / "joined.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(concat_list), "-c", "copy", str(joined)],
        check=True, capture_output=True, text=True,
    )

    shots = detect_shot_boundaries(joined, out_json=tmp_path / "shots.json")
    assert len(shots) >= 2, f"应检出至少 2 个 shot,实际 {shots}"
    # 边界应落在两段交界(2s)附近,且区间首尾覆盖全片
    assert shots[0][0] == 0.0
    assert abs(shots[-1][1] - 4.0) < 0.3
    boundary = shots[1][0]
    assert abs(boundary - 2.0) < 0.5, f"边界应在 2s 附近,实际 {boundary}"
    # 中间产物 shots.json 已写盘且与返回值一致
    import json
    saved = json.loads((tmp_path / "shots.json").read_text(encoding="utf-8"))
    assert saved["n_shots"] == len(shots)
    assert saved["shots"] == [[s, e] for s, e in shots]
