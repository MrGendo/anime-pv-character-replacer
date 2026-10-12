# W3 scope — 替换合成 CLI:python -m anime_pv_replacer.compose(读 W2 轨迹 → 逐帧替换 → ffmpeg 编码导出)。
"""管线阶段 CLI(人物替换与合成导出)。

用法::

    uv run python -m anime_pv_replacer.compose --config config/pipeline.yaml --segment A --method sprite
    uv run python -m anime_pv_replacer.compose --config config/pipeline.yaml --segment B --method palette \\
        --track-id 6 --chara-sheet assets/character_sheets/chara_jun_sheet.png

流程(契约条款 2):读最新一次 W2 tracks.json(--track-id 指定轨迹,默认该片段
最长 track)→ ffmpeg rawvideo 解码逐帧 → replacer 替换 → 逐帧图存
``output/{run_id}/replace/{SEG}_{method}/frames/`` → ffmpeg 编码 final.mp4
(crf/preset 取 config export 段)→ 每 15 帧一张「原帧|替换帧」对比图存
compare_viz/ → replace_summary.json。api_doc 路径未授权时人话提示并退出码 2,
已授权也只打印将要调用的 endpoint/参数,绝不真实调用付费 API(契约条款 4)。
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .config import ConfigError, load_config, resolve_api_key
from .logging_utils import get_run_logger
from .replacement import PaletteReplacer, SpriteReplacer, stabilize_widths
from .segment import new_run_id
from .segmentation import MediaToolError, _require_tool, probe_video

# 对比图抽帧间隔(契约:每 15 帧一张)
COMPARE_EVERY_N_FRAMES = 15
# 逐帧图与对比图的 JPEG 质量(与 W2 可视化产物保持一致)
JPEG_QUALITY = 90


def _iter_decode(video: Path, width: int, height: int):
    """ffmpeg rawvideo 解码生成器,逐帧 yield BGR ndarray。

    用 ffmpeg 而非 OpenCV 解码(契约条款 2):rawvideo 管道输出与编码端
    输入天然同构,帧尺寸/像素格式完全可控。
    """
    ffmpeg = _require_tool("ffmpeg")
    proc = subprocess.Popen(
        [ffmpeg, "-loglevel", "error", "-i", str(video),
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE,
    )
    frame_size = width * height * 3
    try:
        while True:
            buf = proc.stdout.read(frame_size)
            if not buf or len(buf) < frame_size:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def _start_encoder(dst: Path, width: int, height: int, fps: float,
                   crf: int, preset: str) -> subprocess.Popen:
    """起一个吃 rawvideo 帧的 ffmpeg 编码子进程(libx264)。"""
    ffmpeg = _require_tool("ffmpeg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{width}x{height}", "-framerate", f"{fps:.6f}", "-i", "-",
         "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
         "-pix_fmt", "yuv420p", str(dst)],
        stdin=subprocess.PIPE,
    )


def _latest_tracks_artifact(cfg: dict, seg: str) -> Path:
    """定位最新一次 track 产物(与 track.py 的 detect 产物定位同一套路)。"""
    output_root = Path(cfg["paths"]["output_root"])
    candidates = sorted(output_root.glob(f"*/tracking/{seg}/tracks.json"))
    if not candidates:
        raise FileNotFoundError(
            f"找不到片段 {seg} 的 track 产物(搜索 {output_root}/*/tracking/{seg}/),"
            f"请先运行 track --stage detect / --stage track。"
        )
    return candidates[-1]


def _select_track(tracks: list[dict], track_id: int | None) -> dict:
    """按 --track-id 选轨迹;未指定时取最长(帧跨度最大)的那条。"""
    if track_id is not None:
        for t in tracks:
            if t["track_id"] == track_id:
                return t
        raise ValueError(
            f"--track-id {track_id} 不存在,可选:{sorted(t['track_id'] for t in tracks)}"
        )
    return max(tracks, key=lambda t: t["frame_end"] - t["frame_start"])


def _print_api_doc_plan(cfg: dict) -> int:
    """api_doc 已授权时的「只打印不真调」路径(契约条款 4)。"""
    api = cfg["replacement"]["api"]
    print("[api_doc] 付费路径已授权,按契约本仓不真实调用。将要请求:")
    print(f"  POST {api['endpoint']}")
    print(f"  model={api['model'] or '(服务端默认)'} retries={api['retries']} "
          f"timeout_s={api['timeout_s']}")
    print("  输入:目标帧 bbox 区域 + 人设图;输出:风格化替换图(细节见 docs/MATERIALS.md)")
    return 0


def run_replace_stage(cfg: dict, run_id: str, out_root: Path, seg: str, method: str,
                      track_id: int | None, sheet_arg: str | None) -> Path:
    """跑完「解码 → 逐帧替换 → 编码 → 落盘产物」全流程,返回产物目录。"""
    logger = get_run_logger(run_id, out_root / "logs", stage=f"replace-{seg}-{method}")

    video = Path(cfg["paths"]["assets_root"]) / "segments" / f"{seg}.mp4"
    if not video.is_file():
        raise FileNotFoundError(f"片段视频不存在:{video.resolve()}\n请先运行 segment --stage cut 生成片段。")
    info = probe_video(video)
    width, height = info["width"], info["height"]

    tracks_path = _latest_tracks_artifact(cfg, seg)
    data = json.loads(tracks_path.read_text(encoding="utf-8"))
    tracks = data.get("tracks") or []
    if not tracks:
        raise FileNotFoundError(f"{tracks_path} 里没有任何轨迹,请先跑 detect/track 阶段。")
    track = _select_track(tracks, track_id)
    bbox_by_frame = {int(f["frame_idx"]): f["bbox"] for f in track["frames"]}

    rcfg = cfg["replacement"]
    sheet = Path(sheet_arg or rcfg.get("chara_sheet") or "assets/character_sheets/chara_aoi_sheet.png")
    if not sheet.is_file():
        raise FileNotFoundError(f"人设图不存在:{sheet.resolve()}")

    # 时序稳定(契约条款 3):预计算 sprite 宽度序列并做滑动中值滤波
    width_by_frame: dict[int, float] = {}
    n_pulled = 0
    if method == "sprite":
        replacer = SpriteReplacer(sheet, rcfg["edge_blend_px"], rcfg["feather_sigma"])
        idxs = sorted(bbox_by_frame)
        raw = [float(replacer.target_size(bbox_by_frame[i])[0]) for i in idxs]
        widths = stabilize_widths(raw)
        width_by_frame = dict(zip(idxs, widths))
        n_pulled = sum(1 for a, b in zip(raw, widths) if a != b)
    else:
        replacer = PaletteReplacer(sheet)

    out_dir = out_root / "replace" / f"{seg}_{method}"
    frames_dir, viz_dir = out_dir / "frames", out_dir / "compare_viz"
    frames_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    logger.event("replace.start", segment=seg, method=method, video=str(video),
                 tracks_json=str(tracks_path), track_id=track["track_id"],
                 track_frames=[track["frame_start"], track["frame_end"]],
                 chara_sheet=str(sheet), n_frames=info["n_frames"],
                 width_stabilized_pulled=n_pulled)

    final_mp4 = out_dir / "final.mp4"
    encoder = _start_encoder(final_mp4, width, height, info["fps"],
                             cfg["export"]["crf"], cfg["export"]["preset"])
    replace_ms: list[float] = []
    n_frames = n_replaced = 0
    t_start = time.perf_counter()
    try:
        for frame in _iter_decode(video, width, height):
            # 对比帧需要替换前的原图,只在抽帧帧上复制省内存
            orig = frame.copy() if n_frames % COMPARE_EVERY_N_FRAMES == 0 else None
            bbox = bbox_by_frame.get(n_frames)
            if bbox is not None:
                t0 = time.perf_counter()
                frame = replacer.replace(frame, bbox, **({"target_w": width_by_frame[n_frames]}
                                                         if method == "sprite" else {}))
                replace_ms.append((time.perf_counter() - t0) * 1000.0)
                n_replaced += 1
            cv2.imwrite(str(frames_dir / f"{n_frames:06d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            encoder.stdin.write(frame.tobytes())
            if orig is not None:
                cv2.imwrite(str(viz_dir / f"{n_frames:06d}.jpg"),
                            np.hstack([orig, frame]),
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            n_frames += 1
    finally:
        encoder.stdin.close()
        encoder.wait()
    if n_frames == 0:
        raise MediaToolError(f"视频一帧都没读到(文件损坏?):{video}")
    if encoder.returncode != 0:
        raise MediaToolError(f"ffmpeg 编码失败(exit={encoder.returncode}):{final_mp4}")
    total_s = time.perf_counter() - t_start

    summary = {
        "segment": seg,
        "method": method,
        "run_id": run_id,
        "track_id": track["track_id"],
        "track_frames": [track["frame_start"], track["frame_end"]],
        "chara_sheet": str(sheet),
        "tracks_source": str(tracks_path),
        "n_frames": n_frames,
        "n_replaced_frames": n_replaced,
        "avg_replace_ms": round(sum(replace_ms) / len(replace_ms), 2) if replace_ms else 0.0,
        "max_replace_ms": round(max(replace_ms), 2) if replace_ms else 0.0,
        "total_s": round(total_s, 2),
        "final_mp4": str(final_mp4),
        "final_mp4_bytes": final_mp4.stat().st_size,
        "frames_dir": str(frames_dir),
        "compare_viz_dir": str(viz_dir),
        "width_stabilized_pulled": n_pulled,
    }
    with open(out_dir / "replace_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.event("replace.done", segment=seg, method=method,
                 track_id=track["track_id"], n_frames=n_frames,
                 n_replaced_frames=n_replaced,
                 avg_replace_ms=summary["avg_replace_ms"], total_s=summary["total_s"],
                 final_mp4=str(final_mp4), final_mp4_bytes=summary["final_mp4_bytes"],
                 width_stabilized_pulled=n_pulled)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    """CLI 入口,返回退出码(0 成功 / 1 失败 / 2 api_doc 未授权)。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="anime_pv_replacer.compose",
        description="动画 PV 人物替换管线 — 替换合成与导出(W3)",
    )
    parser.add_argument("--config", required=True, help="配置文件路径(YAML)")
    parser.add_argument("--segment", required=True, choices=["A", "B"],
                        help="目标片段名(对应 config.segments 里的 name)")
    parser.add_argument("--method", required=True, choices=["sprite", "palette", "api_doc"],
                        help="替换路径:sprite=人设抠像贴入 / palette=LAB 色板迁移 / api_doc=付费 API(只打印不真调)")
    parser.add_argument("--track-id", type=int, default=None,
                        help="要替换的轨迹 id(默认该片段最长 track)")
    parser.add_argument("--chara-sheet", default=None,
                        help="人设图路径(默认取 config replacement.chara_sheet)")
    args = parser.parse_args(argv)

    run_id = new_run_id()
    try:
        cfg = load_config(args.config)
        if args.method == "api_doc":
            # 契约条款 4:未授权人话提示退出码 2;授权了也只打印 endpoint/参数,不真调
            if not resolve_api_key(cfg):
                print("[失败] 付费路径未授权:未设置 ARK_API_KEY,api_doc 替换不运行。"
                      "授权与费用说明见 docs/MATERIALS.md。", file=sys.stderr)
                return 2
            return _print_api_doc_plan(cfg)
        out_root = Path(cfg["paths"]["output_root"]) / run_id
        run_replace_stage(cfg, run_id, out_root, args.segment, args.method,
                          args.track_id, args.chara_sheet)
    except (ConfigError, FileNotFoundError, MediaToolError, ValueError, KeyError) as e:
        # 契约沿用 W1/W2:外部依赖/前置产物问题只给人话提示,不甩 traceback
        print(f"[失败] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
