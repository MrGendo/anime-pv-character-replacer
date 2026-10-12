# W2 scope — 检测/跟踪 CLI 入口:python -m anime_pv_replacer.track。
"""管线阶段 CLI(人物检测与跟踪)。

用法::

    uv run python -m anime_pv_replacer.track --config config/pipeline.yaml --segment A --stage detect
    uv run python -m anime_pv_replacer.track --config config/pipeline.yaml --segment A --stage track

detect 产物落 ``output/{run_id}/detection/{SEG}/``(逐帧 frames/*.json + 汇总
detections.json);track 自动读取「最新一次 detect 产物」跑跟踪,落
``output/{run_id}/tracking/{SEG}/``(tracks.json + tracks_viz/ 每 10 帧一张
可视化 jpg)。每次执行生成独立 run_id,互不覆盖。
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import sys
import time
from pathlib import Path

import cv2

from .config import ConfigError, load_config
from .detection import DetectionError, PersonDetector
from .logging_utils import get_run_logger
from .segment import new_run_id
from .segmentation import MediaToolError, probe_video
from .tracking import SimpleTracker, iou

# 可视化抽帧间隔(契约:每隔 N=10 帧存一张)
VIZ_EVERY_N_FRAMES = 10
# 可视化配色调色板(BGR),按 track_id 取模循环
VIZ_COLORS = [(62, 76, 240), (140, 200, 70), (70, 140, 255), (215, 130, 60),
              (230, 175, 170), (150, 105, 250), (90, 220, 220)]


def _load_frame_detections(det_dir: Path, n_frames: int) -> list[list[dict]]:
    """读回 detect 阶段的逐帧产物,返回按帧号对齐的检测列表。"""
    frames: list[list[dict]] = []
    for idx in range(n_frames):
        fp = det_dir / "frames" / f"{idx:06d}.json"
        if not fp.is_file():
            raise FileNotFoundError(f"缺逐帧检测产物:{fp}\n请对该片段重跑 --stage detect。")
        frames.append(json.loads(fp.read_text(encoding="utf-8"))["detections"])
    return frames


def _interpolate_stride_frames(det_by_idx: dict[int, list[dict]],
                               hit_frames: list[int], n_frames: int) -> dict[int, list[dict]]:
    """frame_stride > 1 时,对跳过帧做相邻两次真实检测的线性插值。

    近似策略:先用 IoU 贪心把前后两次检测配对(同一目标才插值),
    配对框 bbox/score 线性过渡;只有单侧存在的框按该侧原样保留。
    """
    out: dict[int, list[dict]] = {}
    for i in range(n_frames):
        if i in det_by_idx:
            out[i] = det_by_idx[i]
            continue
        pos = bisect.bisect_left(hit_frames, i)
        prev_k = hit_frames[pos - 1] if pos > 0 else None
        next_k = hit_frames[pos] if pos < len(hit_frames) else None
        prev_d = det_by_idx.get(prev_k, []) if prev_k is not None else []
        next_d = det_by_idx.get(next_k, []) if next_k is not None else []
        if not prev_d and not next_d:
            out[i] = []
            continue
        span = (next_k - prev_k) if (prev_k is not None and next_k is not None) else 1
        t = ((i - prev_k) / span) if span > 0 else 1.0

        pairs: list[tuple[float, int, int]] = []
        for pi, p in enumerate(prev_d):
            for ni, n in enumerate(next_d):
                ov = iou(p["bbox"], n["bbox"])
                if ov > 0.0:
                    pairs.append((ov, pi, ni))
        pairs.sort(key=lambda x: x[0], reverse=True)
        used_p: set[int] = set()
        used_n: set[int] = set()
        dets: list[dict] = []
        for _, pi, ni in pairs:
            if pi in used_p or ni in used_n:
                continue
            used_p.add(pi)
            used_n.add(ni)
            p_box, n_box = prev_d[pi]["bbox"], next_d[ni]["bbox"]
            dets.append({
                "bbox": [round(p_box[c] + (n_box[c] - p_box[c]) * t, 1) for c in range(4)],
                "score": round(prev_d[pi]["score"] + (next_d[ni]["score"] - prev_d[pi]["score"]) * t, 4),
            })
        for pi, p in enumerate(prev_d):   # 仅前侧有:原样保留
            if pi not in used_p:
                dets.append({"bbox": list(p["bbox"]), "score": p["score"]})
        for ni, n in enumerate(next_d):   # 仅后侧有:原样保留
            if ni not in used_n:
                dets.append({"bbox": list(n["bbox"]), "score": n["score"]})
        out[i] = dets
    return out


def run_detect_stage(cfg: dict, run_id: str, out_root: Path, seg: str) -> None:
    """对 assets/segments/{seg}.mp4 逐帧检测,写逐帧 json + 汇总 detections.json。"""
    logger = get_run_logger(run_id, out_root / "logs", stage=f"detect-{seg}")

    video = Path(cfg["paths"]["assets_root"]) / "segments" / f"{seg}.mp4"
    if not video.is_file():
        raise FileNotFoundError(f"片段视频不存在:{video.resolve()}\n请先运行 segment --stage cut 生成片段。")

    info = probe_video(video)  # 借道 ffprobe 取 fps/分辨率;缺 ffmpeg 时其报错自带安装提示
    dcfg = cfg["detection"]
    stride = max(1, int(dcfg.get("frame_stride", 1) or 1))
    detector = PersonDetector(
        method=dcfg["method"],
        score_thresh=float(dcfg["score_thresh"]),
        device=dcfg["device"],
        max_persons=int(dcfg["max_persons"]),
    )
    logger.event("detect.start", segment=seg, video=str(video), method=detector.method,
                 requested_method=dcfg["method"], n_frames=info["n_frames"],
                 fps=info["fps"], frame_stride=stride)

    frames_dir = out_root / "detection" / seg / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise MediaToolError(f"OpenCV 打不开视频(编码不受支持或文件损坏):{video}")

    det_by_idx: dict[int, list[dict]] = {}   # 真实推理帧的检测结果
    infer_ms: list[float] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            t0 = time.perf_counter()
            det_by_idx[frame_idx] = detector.detect(frame)
            infer_ms.append((time.perf_counter() - t0) * 1000.0)
        frame_idx += 1
    cap.release()
    if frame_idx == 0:
        raise MediaToolError(f"视频一帧都没读到(文件损坏?):{video}")

    if stride > 1:  # 跳过帧补齐(契约条款 4)
        det_by_idx = _interpolate_stride_frames(det_by_idx, sorted(det_by_idx), frame_idx)

    per_frame: list[dict] = []
    hist: dict[str, int] = {}
    for idx in range(frame_idx):
        dets = det_by_idx[idx]
        fp = frames_dir / f"{idx:06d}.json"
        with open(fp, "w", encoding="utf-8") as f:
            json.dump({"frame_idx": idx, "detections": dets,
                       "interpolated": stride > 1 and idx % stride != 0}, f,
                      ensure_ascii=False, indent=2)
        n = len(dets)
        hist[str(n)] = hist.get(str(n), 0) + 1
        # 检测耗时只记真实推理帧(stride 命中),插值帧无推理记 0
        ms = round(infer_ms[idx // stride], 2) if idx % stride == 0 else 0.0
        per_frame.append({
            "frame_idx": idx,
            "n_persons": n,
            "avg_score": round(sum(d["score"] for d in dets) / n, 4) if dets else 0.0,
            "detect_ms": ms,
        })

    avg_ms = sum(infer_ms) / len(infer_ms) if infer_ms else 0.0

    summary = {
        "segment": seg,
        "source": str(video),
        "run_id": run_id,
        "method": detector.method,
        "score_thresh": detector.score_thresh,
        "max_persons": detector.max_persons,
        "frame_stride": stride,
        "fps": info["fps"],
        "width": info["width"],
        "height": info["height"],
        "n_frames": frame_idx,
        "avg_detect_ms": round(avg_ms, 2),
        "max_detect_ms": round(max(infer_ms), 2) if infer_ms else 0.0,
        "person_count_hist": hist,   # 每帧人数分布 {"人数": 帧数}
        "frames": per_frame,
    }
    out_json = out_root / "detection" / seg / "detections.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.event("detect.done", segment=seg, method=detector.method, n_frames=frame_idx,
                 avg_detect_ms=round(avg_ms, 2), person_count_hist=hist,
                 detections_json=str(out_json))


def _latest_detection_artifact(cfg: dict, seg: str) -> Path:
    """定位最新一次 detect 产物(run_id 为时间戳,目录名字典序即时间序)。"""
    output_root = Path(cfg["paths"]["output_root"])
    candidates = sorted(output_root.glob(f"*/detection/{seg}/detections.json"))
    if not candidates:
        raise FileNotFoundError(
            f"找不到片段 {seg} 的 detect 产物(搜索 {output_root}/*/detection/{seg}/),"
            f"请先运行 --stage detect。"
        )
    return candidates[-1]


def run_track_stage(cfg: dict, run_id: str, out_root: Path, seg: str) -> None:
    """读最新 detect 产物跑跟踪,写 tracks.json + tracks_viz/ 可视化。"""
    logger = get_run_logger(run_id, out_root / "logs", stage=f"track-{seg}")

    det_json_path = _latest_detection_artifact(cfg, seg)
    meta = json.loads(det_json_path.read_text(encoding="utf-8"))
    n_frames = int(meta["n_frames"])
    logger.event("track.read_detections", segment=seg, source=str(det_json_path),
                 n_frames=n_frames, method=meta["method"])

    frame_dets = _load_frame_detections(det_json_path.parent, n_frames)
    tcfg = cfg["tracking"]
    tracker = SimpleTracker(
        iou_threshold=float(tcfg.get("iou_threshold", 0.3)),
        max_gap_frames=int(tcfg["max_gap_frames"]),
        smoothing_window=int(tcfg["smoothing_window"]),
    )
    per_frame_returns = [
        tracker.update(frame_dets[i], i) for i in range(n_frames)
    ]
    tracks = tracker.export_tracks()
    longest = max((t["frame_end"] - t["frame_start"] + 1 for t in tracks), default=0)

    track_out = out_root / "tracking" / seg
    track_out.mkdir(parents=True, exist_ok=True)
    tracks_json = track_out / "tracks.json"
    with open(tracks_json, "w", encoding="utf-8") as f:
        json.dump({
            "segment": seg,
            "detection_source": str(det_json_path),
            "config": {k: tcfg.get(k, v) for k, v in
                       [("iou_threshold", 0.3), ("max_gap_frames", 8), ("smoothing_window", 7)]},
            "n_frames": n_frames,
            "n_tracks": len(tracks),
            "longest_track_frames": longest,
            "tracks": tracks,
        }, f, ensure_ascii=False, indent=2)

    # 可视化:每 VIZ_EVERY_N_FRAMES 帧重读原视频画 bbox/track_id(顺序读避免按帧 seek 的误差)
    viz_dir = track_out / "tracks_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(meta["source"])
    viz_paths: list[str] = []
    for idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if idx % VIZ_EVERY_N_FRAMES != 0:
            continue
        for item in per_frame_returns[idx]:
            if item["state"] == "LOST":   # 空缺帧不画框,如实反映检测空档
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in item["bbox"])
            color = VIZ_COLORS[item["track_id"] % len(VIZ_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"id{item['track_id']}", (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        jpg = viz_dir / f"{idx:06d}.jpg"
        cv2.imwrite(str(jpg), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        viz_paths.append(str(jpg))
    cap.release()

    logger.event("track.done", segment=seg, n_tracks=len(tracks),
                 longest_track_frames=longest, tracks_json=str(tracks_json),
                 n_viz=len(viz_paths), viz_dir=str(viz_dir))


def main(argv: list[str] | None = None) -> int:
    """CLI 入口,返回进程退出码(0 成功 / 1 失败)。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="anime_pv_replacer.track",
        description="动画 PV 人物替换管线 — 人物检测与跟踪(W2)",
    )
    parser.add_argument("--config", required=True, help="配置文件路径(YAML)")
    parser.add_argument("--segment", required=True, choices=["A", "B"],
                        help="目标片段名(对应 config.segments 里的 name)")
    parser.add_argument("--stage", required=True, choices=["detect", "track"],
                        help="执行阶段:detect=逐帧人物检测 / track=IoU 跟踪")
    args = parser.parse_args(argv)

    run_id = new_run_id()
    try:
        cfg = load_config(args.config)
        out_root = Path(cfg["paths"]["output_root"]) / run_id
        if args.stage == "detect":
            run_detect_stage(cfg, run_id, out_root, args.segment)
        else:
            run_track_stage(cfg, run_id, out_root, args.segment)
    except (ConfigError, FileNotFoundError, MediaToolError, DetectionError, ValueError) as e:
        # 契约条款 6:外部依赖/前置产物问题只给人话提示,不甩 traceback
        print(f"[失败] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
