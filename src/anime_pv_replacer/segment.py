# W1 scope — 切片/镜头检测 CLI 入口:python -m anime_pv_replacer.segment。
"""管线阶段 CLI。

用法::

    uv run python -m anime_pv_replacer.segment --config config/pipeline.yaml --stage cut
    uv run python -m anime_pv_replacer.segment --config config/pipeline.yaml --stage shots

每次执行生成 run_id(YYYYmmdd_HHMMSS):cut 的媒体产物按配置进 assets/segments/,
shots 产物与运行日志统一落在 output/{run_id}/ 下,互不覆盖,便于回溯比对。
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from .config import ConfigError, load_config
from .logging_utils import get_run_logger
from .segmentation import (
    MediaToolError,
    cut_segment,
    detect_shot_boundaries,
    probe_video,
)

# run 目录约定:output/{run_id}/{stage_dir}/;shots 阶段按契约固定叫 detection
STAGE_DIRS = {"cut": "segments", "shots": "detection"}


def new_run_id() -> str:
    """生成 YYYYmmdd_HHMMSS 格式的 run 标识。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_cut_stage(cfg: dict, run_id: str, out_root: Path) -> None:
    """按 config.segments 逐段切片到 {assets_root}/segments/。"""
    logger = get_run_logger(run_id, out_root / "logs", stage="cut")

    video = Path(cfg["source"]["video_path"])
    if not video.is_file():
        raise FileNotFoundError(
            f"源视频不存在:{video.resolve()}\n"
            f"请检查 config 中 source.video_path 的配置。"
        )

    seg_dir = Path(cfg["paths"]["assets_root"]) / "segments"
    logger.event("cut.start", video=str(video), n_segments=len(cfg["segments"]))

    for seg in cfg["segments"]:
        dst, actual = cut_segment(video, float(seg["start"]), float(seg["end"]),
                                  seg_dir / f"{seg['name']}.mp4")
        logger.event("cut.segment_done", name=seg["name"],
                     dst=str(dst), expected_s=seg["end"] - seg["start"],
                     actual_s=round(actual, 3))

    logger.event("cut.done")


def run_shots_stage(cfg: dict, run_id: str, out_root: Path) -> None:
    """对源视频跑镜头边界检测,写 output/{run_id}/detection/shots.json。"""
    logger = get_run_logger(run_id, out_root / "logs", stage="shots")

    video = Path(cfg["source"]["video_path"])
    if not video.is_file():
        raise FileNotFoundError(
            f"源视频不存在:{video.resolve()}\n"
            f"请检查 config 中 source.video_path 的配置。"
        )

    info = probe_video(video)
    logger.event("shots.start", video=str(video),
                 duration_s=info["duration_s"], fps=info["fps"])

    shots_json = out_root / STAGE_DIRS["shots"] / "shots.json"
    shots = detect_shot_boundaries(video, out_json=shots_json)
    logger.event("shots.done", n_shots=len(shots), shots_json=str(shots_json),
                 first_shots=[list(s) for s in shots[:5]])


def main(argv: list[str] | None = None) -> int:
    """CLI 入口,返回进程退出码(0 成功 / 1 失败)。"""
    # 只上 INFO 以上:让 segmentation 内部的 warning(如切片超差)直接可见
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="anime_pv_replacer.segment",
        description="动画 PV 人物替换管线 — 切片与镜头边界检测(W1)",
    )
    parser.add_argument("--config", required=True, help="配置文件路径(YAML)")
    parser.add_argument("--stage", required=True, choices=["cut", "shots"],
                        help="执行阶段:cut=按 segments 切片 / shots=镜头边界检测")
    args = parser.parse_args(argv)

    run_id = new_run_id()
    try:
        cfg = load_config(args.config)
        out_root = Path(cfg["paths"]["output_root"]) / run_id
        if args.stage == "cut":
            run_cut_stage(cfg, run_id, out_root)
        else:
            run_shots_stage(cfg, run_id, out_root)
    except (ConfigError, FileNotFoundError, MediaToolError, ValueError) as e:
        # 契约条款 5:外部依赖/配置问题只给人话提示,不甩 traceback
        print(f"[失败] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
