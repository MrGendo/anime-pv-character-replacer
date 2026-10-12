# W4 scope — 评估 CLI:扫 run-dir 的 replace 组合 → 五维指标 + 泛化/成本 → experiments/ JSON+MD 报告。
"""管线阶段 CLI(自动化评估与报告产物,契约条款 2)。

用法::

    uv run python -m anime_pv_replacer.evaluate --config config/pipeline.yaml --run-dir output/{run_id}
    uv run python -m anime_pv_replacer.evaluate --config config/pipeline.yaml --run-dir output/{run_id} \\
        --blind experiments/blind_review/review.csv --out-dir experiments

流程:扫 ``{run-dir}/replace/{SEG}_{method}/``(以 replace_summary.json 为准)→
按片段把原视频重解码为 JPEG q90 逐帧图(与 W3 frames/ 同编码,比较两侧退化
一致)→ 逐组合算五维指标(evaluation.py)→ 分段线性映射 + 按
docs/EVALUATION_DESIGN.md 权重加总 → experiments/eval_{run_id}.json + .md。
--blind CSV(columns: segment,method,dim,score,comment)存在时并入 md 单独
一节,不混入自动分。原始片段缺失/轨迹缺失的组合指标记 N/A 并继续,不中断
整轮(契约条款 1 的整轮不中断要求)。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .compose import _iter_decode
from .config import ConfigError, load_config
from .evaluation import (
    _clip_roi,
    background_fidelity,
    clip_identity_score,
    flicker_score,
    motion_consistency,
    piecewise_score,
    quality_score,
    temporal_flicker,
)
from .logging_utils import get_run_logger
from .segmentation import MediaToolError, probe_video

logger = logging.getLogger(__name__)

# 维度权重按 docs/EVALUATION_DESIGN.md(100 分制);config evaluation.weights 是
# W1 占位方案(ssim/lpips/clip_i),与本题指标体系不一致,不采用
DIM_SPECS = [
    ("identity", "人物身份一致性", 0.25),
    ("quality", "画面质量", 0.15),
    ("motion", "动作一致性", 0.20),
    ("temporal", "时序稳定性", 0.20),
    ("background", "背景保真度", 0.10),
]
GEN_WEIGHT = 0.10           # 跨片段泛化权重(五维合计 0.9 + 泛化 0.1 = 1.0)
GEN_SPREAD_PENALTY = 25.0   # A/B 五维小计差达到此值(90 分制)时一致性项归零
METHOD_ORDER = ("sprite", "palette")
JPEG_QUALITY = 90           # 原始帧重解码落盘质量,与 W3 compose 的 frames/ 一致


def _find_combos(run_dir: Path) -> list[Path]:
    """run-dir 下全部合法组合目录(replace/*且有 replace_summary.json)。

    排序:片段名升序,方法按 sprite→palette(报告列顺序稳定)。
    """
    replace_root = run_dir / "replace"
    if not replace_root.is_dir():
        return []
    combos = [d for d in replace_root.iterdir()
              if d.is_dir() and (d / "replace_summary.json").is_file()]
    return sorted(combos, key=lambda d: (d.name.split("_")[0],
                                         METHOD_ORDER.index(d.name.split("_")[-1])
                                         if d.name.split("_")[-1] in METHOD_ORDER else 99))


def _decode_orig_frames(video: Path, dst: Path) -> int:
    """原片段重解码为 JPEG q90 逐帧图,返回帧数(供目录级指标消费)。"""
    info = probe_video(video)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for frame in _iter_decode(video, info["width"], info["height"]):
        cv2.imwrite(str(dst / f"{n:06d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        n += 1
    return n


def _load_bboxes(summary: dict) -> dict[int, list[float]]:
    """从 replace_summary 指向的 tracks.json 取替换轨迹的逐帧 bbox。"""
    tracks_path = Path(summary.get("tracks_source", ""))
    if not tracks_path.is_file():
        logger.warning("tracks.json 不存在:%s(组合指标将记 N/A)", tracks_path)
        return {}
    data = json.loads(tracks_path.read_text(encoding="utf-8"))
    track = next((t for t in data.get("tracks", [])
                  if t.get("track_id") == summary.get("track_id")), None)
    if track is None:
        logger.warning("轨迹 %s 在 %s 里不存在", summary.get("track_id"), tracks_path)
        return {}
    return {int(f["frame_idx"]): f["bbox"] for f in track["frames"]}


def _mean_or_none(values: list[float]):
    """均值;空列表返回 None(维度记 N/A 而不是中断)。"""
    return round(float(np.mean(values)), 4) if values else None


def _evaluate_combo(cfg: dict, combo_dir: Path, tmp_root: Path,
                    orig_dirs: dict[str, Path | None]) -> dict:
    """评估单个组合目录,返回 metrics/cost 齐全的结果 dict(不抛数据类异常)。"""
    summary = json.loads((combo_dir / "replace_summary.json").read_text(encoding="utf-8"))
    seg, method = summary["segment"], summary["method"]
    result: dict = {
        "label": combo_dir.name, "segment": seg, "method": method,
        "combo_dir": str(combo_dir), "note": "",
        "metrics": {"identity": None, "quality": None, "motion": None,
                    "temporal": None, "background": None},
        "frames_scored": {"identity": 0, "quality": 0, "background": 0},
    }
    bboxes = _load_bboxes(summary)
    n_track = len(bboxes)
    n_replaced = int(summary.get("n_replaced_frames", 0))
    result["cost"] = {
        # 成本记录(契约条款 2):每帧耗时/失败帧/干预,全部来自 replace_summary
        "avg_replace_ms": summary.get("avg_replace_ms", 0.0),
        "max_replace_ms": summary.get("max_replace_ms", 0.0),
        "total_s": summary.get("total_s", 0.0),
        "n_frames": summary.get("n_frames", 0),
        "n_replaced_frames": n_replaced,
        "n_track_frames": n_track,
        "failed_frames": max(0, n_track - n_replaced),
        "width_stabilized_pulled": summary.get("width_stabilized_pulled", 0),
        "manual_interventions": summary.get("manual_interventions", 0),
        "api_cost_cny": 0.0,  # sprite/palette 均为本地路径
        "completion": round(n_replaced / n_track, 4) if n_track else None,
    }
    if not bboxes:
        result["note"] = "轨迹缺失,全部指标 N/A"
        return result

    # 原片段按需重解码一次(同片段多组合共享);失败则该组合记 N/A
    if seg not in orig_dirs:
        video = Path(cfg["paths"]["assets_root"]) / "segments" / f"{seg}.mp4"
        if not video.is_file():
            logger.warning("原始片段不存在:%s(组合 %s 记 N/A)", video, combo_dir.name)
            orig_dirs[seg] = None
        else:
            orig_dirs[seg] = tmp_root / seg
            _decode_orig_frames(video, orig_dirs[seg])
    orig_dir = orig_dirs[seg]
    if orig_dir is None:
        result["note"] = "原始片段视频缺失,全部指标 N/A"
        return result

    frames_dir = combo_dir / "frames"
    sheet = summary.get("chara_sheet", "")
    acc: dict[str, list[float]] = {"identity": [], "quality": [], "background": []}
    for idx in sorted(bboxes):
        o = cv2.imread(str(orig_dir / f"{idx:06d}.jpg"), cv2.IMREAD_COLOR)
        r = cv2.imread(str(frames_dir / f"{idx:06d}.jpg"), cv2.IMREAD_COLOR)
        if o is None or r is None or o.shape != r.shape:
            logger.warning("组合 %s 帧 %d 原始/替换图缺失或尺寸不符,跳过", combo_dir.name, idx)
            continue
        bbox = bboxes[idx]
        roi = _clip_roi(r.shape, bbox)
        if roi is None:
            continue
        x1, y1, x2, y2 = roi
        v = clip_identity_score(o, r, sheet, bbox)
        if v is not None:
            acc["identity"].append(v)
        # 画质只看替换区域(全帧会被未动的背景稀释)
        v = quality_score(r[y1:y2, x1:x2], o[y1:y2, x1:x2])
        if v is not None:
            acc["quality"].append(v)
        v = background_fidelity(o, r, bbox)
        if v is not None:
            acc["background"].append(v)
    result["metrics"].update({k: _mean_or_none(v) for k, v in acc.items()})
    result["frames_scored"] = {k: len(v) for k, v in acc.items()}
    result["metrics"]["motion"] = motion_consistency(orig_dir, frames_dir, bboxes)
    result["metrics"]["temporal"] = temporal_flicker(orig_dir, frames_dir, bboxes)
    return result


def _score_results(results: list[dict]) -> dict:
    """就地打分:原始值 → 映射分 → 五维小计(满 90)→ 按方法算泛化 → 总分。"""
    for r in results:
        scores = {}
        for key, _, _ in DIM_SPECS:
            raw = r["metrics"].get(key)
            if raw is None:
                scores[key] = None
            elif key == "temporal":
                scores[key] = flicker_score(raw)  # 闪烁比反向映射
            else:
                scores[key] = piecewise_score(raw)
        r["scores"] = scores
        active = [(w, scores[key]) for key, _, w in DIM_SPECS if scores[key] is not None]
        wsum = sum(w for w, _ in active)
        # 缺维度按剩余权重归一,小计保持「满 90」刻度可比
        r["subtotal"] = round(sum(s * w for w, s in active) * 0.9 / wsum, 2) if wsum else None

    gen: dict[str, dict] = {}
    for method in METHOD_ORDER:
        rs = [r for r in results if r["method"] == method and r["subtotal"] is not None]
        entry = {"segments": [r["segment"] for r in rs],
                 "subtotals": [r["subtotal"] for r in rs]}
        if len(rs) >= 2:
            spread = round(abs(rs[0]["subtotal"] - rs[1]["subtotal"]), 2)
            consistency = 1.0 - min(1.0, spread / GEN_SPREAD_PENALTY)
        else:
            spread, consistency = None, None
        entry["subtotal_spread"] = spread
        comps = [r["cost"]["completion"] for r in rs if r["cost"]["completion"] is not None]
        completion = round(float(np.mean(comps)), 4) if comps else None
        entry["completion"] = completion
        # 泛化 = 完成率与跨段一致性的均值(各 0.5),再过分段线性映射
        if consistency is not None and completion is not None:
            entry["score"] = piecewise_score(0.5 * completion + 0.5 * consistency)
        else:
            entry["score"] = None
        # 关键指标方差:同一维度 A/B 映射分的总体标准差(σ across segments)
        dim_std = {}
        for key, dim_name, _ in DIM_SPECS:
            vals = [r["scores"][key] for r in rs if r["scores"][key] is not None]
            dim_std[key] = round(float(np.std(vals)), 2) if len(vals) >= 2 else None
        entry["dim_std"] = dim_std
        gen[method] = entry

    for r in results:
        g = gen.get(r["method"], {}).get("score")
        r["generalization_score"] = g
        # 小计为 0-90 刻度,泛化分 0-100 × 0.1 = 最多 10 分,合计满 100
        r["total"] = (round(r["subtotal"] + g * GEN_WEIGHT, 2)
                      if r["subtotal"] is not None and g is not None else r["subtotal"])
    return gen


def _load_blind(path: str) -> list[dict]:
    """读盲评 CSV(columns: segment,method,dim,score,comment),坏行跳过。"""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            try:
                rows.append({"segment": (row.get("segment") or "").strip(),
                             "method": (row.get("method") or "").strip(),
                             "dim": (row.get("dim") or "").strip(),
                             "score": float(row.get("score") or ""),
                             "comment": (row.get("comment") or "").strip()})
            except (TypeError, ValueError):
                logger.warning("盲评 CSV 第 %d 行 score 非数值,跳过", i + 2)
    if not rows:
        logger.warning("盲评 CSV 没有有效行:%s", path)
    return rows


def _cell(r: dict, key: str) -> str:
    """md 表格单元:原始值 → 映射分;缺数据记 N/A。"""
    raw, score = r["metrics"].get(key), r["scores"].get(key)
    if raw is None or score is None:
        return "N/A"
    return f"{raw:.3f} → {score:.1f}"


def _render_md(run_id: str, run_dir: Path, results: list[dict], gen: dict,
               blind_rows: list[dict] | None) -> str:
    """组装人读报告(md):自动分主表 + 泛化/成本表尾 + 盲评单独一节。"""
    labels = [r["label"] for r in results]
    lines = [
        f"# 替换质量评估报告 — {run_id}", "",
        f"- run-dir:`{run_dir}`;由 `anime_pv_replacer.evaluate` 自动生成,指标与权重见 `docs/EVALUATION_DESIGN.md`。",
        "- 身份维度为 CLIP 不可用时的降级近似(LAB 直方图相关性 + 主色板余弦)。",
        "- 原始帧重解码为 JPEG q90(与 W3 逐帧产物同编码),背景逐像素相等率因此含 JPEG 压缩噪声。",
        "- 时序稳定性原始值为闪烁比(1.0=与原视频一致,>1 闪烁),得分按 |1-比| 反向映射。",
        "", "## 自动指标(100 分制)", "",
        "| 维度 | 权重 | " + " | ".join(labels) + " |",
        "|---|---|" + "---|" * len(labels),
    ]
    for key, dim_name, w in DIM_SPECS:
        lines.append(f"| {dim_name} | {int(w * 100)}% | "
                     + " | ".join(_cell(r, key) for r in results) + " |")
    lines.append("| 五维加权小计(满 90) | 90% | "
                 + " | ".join(f"{r['subtotal']:.1f}" if r["subtotal"] is not None else "N/A"
                             for r in results) + " |")
    lines.append("| 跨片段泛化 | 10% | "
                 + " | ".join(f"{r['generalization_score']:.1f}"
                             if r["generalization_score"] is not None else "N/A"
                             for r in results) + " |")
    lines.append("| **总分(满 100)** | 100% | "
                 + " | ".join(f"**{r['total']:.1f}**" if r["total"] is not None else "N/A"
                             for r in results) + " |")
    if any(r["metrics"].get(k) is None for r in results for k, _, _ in DIM_SPECS):
        lines.append("")
        lines.append("> 注:标 N/A 的维度数据不足(如 2 帧只有 1 个光流对,算不出相关系数),"
                     "小计按剩余维度权重归一,仍为满分 90 刻度。")

    lines += ["", "### 跨片段泛化明细(表尾)", "",
              "| 方法 | 小计差(A-B) | 完成率 | 逐维 σ(identity/quality/motion/temporal/background) |",
              "|---|---|---|---|"]
    for method, g in gen.items():
        spread = f"{g['subtotal_spread']:.1f}" if g["subtotal_spread"] is not None else "N/A"
        comp = f"{g['completion']:.2%}" if g["completion"] is not None else "N/A"
        sig = " / ".join("—" if g["dim_std"][k] is None else f"{g['dim_std'][k]:.1f}"
                         for k, _, _ in DIM_SPECS)
        lines.append(f"| {method} | {spread} | {comp} | {sig} |")

    lines += ["", "## 成本记录(来自各组合 replace_summary.json)", "",
              "| 组合 | 帧数 | 替换帧 | 失败帧 | 每帧耗时 avg/max ms | 总耗时 s | 宽度稳定拉回 | 人工干预 | API 费用 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        c = r["cost"]
        lines.append(
            f"| {r['label']} | {c['n_frames']} | {c['n_replaced_frames']} | {c['failed_frames']} "
            f"| {c['avg_replace_ms']}/{c['max_replace_ms']} | {c['total_s']} "
            f"| {c['width_stabilized_pulled']} | {c['manual_interventions']}(无记录字段) "
            f"| ¥{c['api_cost_cny']:.2f} |")

    if blind_rows:
        lines += ["", "## 盲评(1-5 分,单独列示,不计入自动分)", ""]
        # 汇总表:同组多份评分取均值;逐行明细(含理由)在下方与 JSON 里
        agg: dict[tuple[str, str, str], list[float]] = {}
        for b in blind_rows:
            agg.setdefault((b["segment"], b["method"], b["dim"]), []).append(b["score"])
        lines += ["### 汇总", "", "| 片段 | 方法 | 维度 | 均分 | 样本数 |", "|---|---|---|---|---|"]
        for (seg, method, dim), vals in sorted(agg.items()):
            lines.append(f"| {seg} | {method} | {dim} | {np.mean(vals):.1f} | {len(vals)} |")
        lines += ["", "### 明细", "", "| 片段 | 方法 | 维度 | 评分 | 理由 |", "|---|---|---|---|---|"]
        for b in sorted(blind_rows, key=lambda x: (x["segment"], x["method"], x["dim"])):
            lines.append(f"| {b['segment']} | {b['method']} | {b['dim']} | {b['score']:g} "
                         f"| {b['comment'] or '—'} |")

    lines += ["", "## 计分规则", "",
              "- 分段线性映射:0.0→0 分,0.5→60 分,0.75→80 分,1.0→100 分(时序按 |1-闪烁比| 反向)。",
              f"- 泛化分 = piecewise(0.5×完成率 + 0.5×(1 - 小计差/{int(GEN_SPREAD_PENALTY)})),"
              "再乘 10% 计入总分;单片段方法无泛化分,总分=五维小计。", ""]
    return "\n".join(lines)


def _build_payload(run_id: str, run_dir: Path, config_path: str, results: list[dict],
                   gen: dict, blind_rows: list[dict] | None) -> dict:
    """组装 eval JSON 的完整负载(与 md 同源,供程序化消费)。"""
    return {
        "run_id": run_id,
        "tool": "anime_pv_replacer.evaluate",
        "config": config_path,
        "run_dir": str(run_dir),
        "scoring": {
            "knots": "0→0,0.5→60,0.75→80,1.0→100(temporal 按 |1-比| 反向)",
            "weights": {name: w for _, name, w in DIM_SPECS},
            "generalization_weight": GEN_WEIGHT,
            "generalization_formula":
                f"piecewise(0.5*completion + 0.5*(1 - spread/{GEN_SPREAD_PENALTY}))",
        },
        "combos": results,
        "generalization": gen,
        "blind": blind_rows,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI 入口,返回退出码(0 成功 / 1 失败)。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="anime_pv_replacer.evaluate",
        description="动画 PV 人物替换管线 — 自动化评估与报告产物(W4)",
    )
    parser.add_argument("--config", required=True, help="配置文件路径(YAML)")
    parser.add_argument("--run-dir", required=True,
                        help="要评估的 run 目录(其下 replace/{SEG}_{method}/ 会被扫描)")
    parser.add_argument("--blind", default=None,
                        help="盲评 CSV 路径(columns: segment,method,dim,score,comment;并入 md 单独一节)")
    parser.add_argument("--out-dir", default="experiments", help="报告输出目录(默认 experiments/)")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    try:
        cfg = load_config(args.config)
        combos = _find_combos(run_dir)
        if not combos:
            raise FileNotFoundError(
                f"{run_dir}/replace/ 下没有任何带 replace_summary.json 的组合目录;"
                f"请先运行 compose 生成替换产物,或检查 run-dir 是否正确。"
            )
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = run_dir.name
        rlog = get_run_logger(f"eval_{run_id}", run_dir / "logs", stage="evaluate")
        rlog.event("eval.start", run_dir=str(run_dir), combos=[c.name for c in combos])

        results: list[dict] = []
        with tempfile.TemporaryDirectory(prefix="anime_pv_eval_") as td:
            orig_dirs: dict[str, Path | None] = {}
            for combo_dir in combos:
                try:
                    r = _evaluate_combo(cfg, combo_dir, Path(td), orig_dirs)
                except (FileNotFoundError, MediaToolError, ValueError, KeyError,
                        json.JSONDecodeError, OSError) as e:
                    # 数据类问题只废掉当前组合,不中断整轮(契约条款 1)
                    logger.warning("组合 %s 评估失败:%s", combo_dir.name, e)
                    continue
                results.append(r)
                rlog.event("eval.combo", label=r["label"], metrics=r["metrics"])
        if not results:
            raise FileNotFoundError(f"{run_dir} 下没有任何组合评估成功,见上方 warning。")

        gen = _score_results(results)
        blind_rows = _load_blind(args.blind) if args.blind else None
        payload = _build_payload(run_id, run_dir, args.config, results, gen, blind_rows)
        md = _render_md(run_id, run_dir, results, gen, blind_rows)

        json_path = out_dir / f"eval_{run_id}.json"
        md_path = out_dir / f"eval_{run_id}.md"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(md, encoding="utf-8")
        rlog.event("eval.done", json=str(json_path), md=str(md_path),
                   totals={r["label"]: r["total"] for r in results})

        print("[评估完成] " + " | ".join(
            f"{r['label']} 总分 {r['total']:.1f}" if r["total"] is not None
            else f"{r['label']} 总分 N/A" for r in results))
        print(f"报告:{md_path} / {json_path}")
    except (ConfigError, FileNotFoundError, MediaToolError, ValueError, KeyError) as e:
        # 契约沿用 W1-W3:外部依赖/前置产物问题只给人话提示,不甩 traceback
        print(f"[失败] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
