# W1 scope — 切片与镜头边界:ffprobe 探测、ffmpeg 切片(流复制优先)、HSV 直方图镜头检测。
"""切片与镜头边界检测。

所有媒体信息一律来自 ffprobe 的 JSON 输出解析,不做字符串猜测;
ffmpeg 命令参数以 list 形式传给 subprocess,杜绝 shell 注入与空格路径问题。
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# 外部依赖缺失时的统一人话提示(契约条款 5:不带 traceback 轰炸)
_FFMPEG_INSTALL_HINT = (
    "缺少系统工具 {tool},请先安装( macOS: brew install ffmpeg;"
    " Debian/Ubuntu: sudo apt install ffmpeg)并确认其在 PATH 中。"
)

# 切片时长允许误差(秒):stream copy 按关键帧对齐,超差只记 warning 不判失败
CUT_DURATION_TOLERANCE_S = 0.2


class MediaToolError(Exception):
    """ffmpeg/ffprobe 不可用或调用失败。"""


def _require_tool(tool: str) -> str:
    """确认外部工具存在,返回其可执行路径;缺失时给出安装提示。"""
    path = shutil.which(tool)
    if path is None:
        raise MediaToolError(_FFMPEG_INSTALL_HINT.format(tool=tool))
    return path


def _run_tool(args: list[str]) -> subprocess.CompletedProcess:
    """执行 ffmpeg/ffprobe 子进程,统一超时与错误包装。"""
    try:
        return subprocess.run(
            args, capture_output=True, text=True, timeout=600, check=False
        )
    except FileNotFoundError as e:
        raise MediaToolError(_FFMPEG_INSTALL_HINT.format(tool=args[0])) from e


def probe_video(path: str | Path) -> dict:
    """探测视频元信息,返回 {width, height, fps, n_frames, duration_s, codec}。

    数据全部取自 ``ffprobe -print_format json`` 输出;
    容器无 nb_frames 时(VP9 常见)用 duration_s * fps 估算帧数。
    """
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"视频文件不存在:{src.resolve()}")

    ffprobe = _require_tool("ffprobe")
    proc = _run_tool(
        [ffprobe, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(src)]
    )
    if proc.returncode != 0:
        raise MediaToolError(f"ffprobe 探测失败:{src}\n{proc.stderr.strip()[:500]}")

    info = json.loads(proc.stdout)
    video = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "video"), None
    )
    if video is None:
        raise MediaToolError(f"文件里没有视频流:{src}")

    num, _, den = video.get("r_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den) if float(den) != 0 else 0.0
    duration_s = float(info.get("format", {}).get("duration") or video.get("duration") or 0.0)
    n_frames = int(video.get("nb_frames") or round(duration_s * fps))

    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": round(fps, 6),
        "n_frames": n_frames,
        "duration_s": round(duration_s, 3),
        "codec": video.get("codec_name", ""),
    }


def cut_segment(src: str | Path, start: float, end: float, dst: str | Path) -> tuple[Path, float]:
    """把 ``src`` 的 [start, end) 秒切到 ``dst``,返回 (dst 路径, 实际时长秒)。

    输入定位(-ss)放在 -i 之前做快速 seek;优先 stream copy,
    失败(如关键帧不对齐导致容器不收)自动降级为 libx264 重编码。
    切完后用 ffprobe 复核时长,与期望误差超过 0.2s 记 warning。
    """
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise FileNotFoundError(f"源视频不存在:{src.resolve()}")
    if end <= start:
        raise ValueError(f"切片区间非法:end({end}) 必须大于 start({start})")

    ffmpeg = _require_tool("ffmpeg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start

    # 第一步:stream copy,零转码损耗、速度最快
    copy_args = [ffmpeg, "-y", "-loglevel", "error",
                 "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{duration:.3f}",
                 "-c", "copy", str(dst)]
    proc = _run_tool(copy_args)
    if proc.returncode != 0:
        # 降级:重编码。VP9 源在部分封装下 stream copy 会失败,属预期路径
        logger.warning("stream copy 失败,降级重编码:%s", proc.stderr.strip()[:300])
        reencode_args = [ffmpeg, "-y", "-loglevel", "error",
                         "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{duration:.3f}",
                         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                         "-c:a", "aac", str(dst)]
        proc = _run_tool(reencode_args)
        if proc.returncode != 0:
            raise MediaToolError(f"ffmpeg 切片失败(含重编码降级):{src} -> {dst}\n"
                                 f"{proc.stderr.strip()[:500]}")

    actual = probe_video(dst)["duration_s"]
    if abs(actual - duration) > CUT_DURATION_TOLERANCE_S:
        logger.warning(
            "切片时长超差:%s 期望 %.3fs 实际 %.3fs(容差 %.1fs,stream copy 按关键帧对齐所致)",
            dst, duration, actual, CUT_DURATION_TOLERANCE_S,
        )
    return dst, actual


def detect_shot_boundaries(
    src: str | Path,
    threshold: float = 0.35,
    min_gap_s: float = 1.0,
    out_json: str | Path | None = None,
) -> list[tuple[float, float]]:
    """帧差分直方图法检测镜头边界,返回 [(shot_start_s, shot_end_s), ...]。

    每帧计算 HSV 色彩空间 H/S 二维直方图,与前一帧做 Bhattacharyya 比较,
    距离超过 ``threshold`` 判为候选边界;再按 ``min_gap_s`` 抑制密集误检
    (渐变转场会连续多帧超阈值,只取第一个)。
    帧先缩到宽 320 再算直方图:颜色分布统计对分辨率不敏感,能快约 10 倍。
    提供 ``out_json`` 时把结果连同检测参数写入中间产物 shots.json。
    """
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - 环境缺库时给人话提示
        raise MediaToolError(
            "缺少 opencv-python,请运行 uv sync 安装项目依赖后重试。"
        ) from e

    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(f"视频文件不存在:{src.resolve()}")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise MediaToolError(f"OpenCV 打不开视频(编码不受支持或文件损坏):{src}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:  # nan/0 元数据兜底,以 ffprobe 为准
        fps = probe_video(src)["fps"]
    min_gap_frames = max(1, round(min_gap_s * fps))

    def frame_hist(frame):
        """缩放 + HSV 直方图,输出与帧分辨率无关的颜色分布描述。"""
        small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    boundaries: list[int] = []  # 每个新镜头的首帧序号
    prev_hist = None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hist = frame_hist(frame)
        if prev_hist is not None:
            dist = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if dist > threshold and (
                not boundaries or frame_idx - boundaries[-1] >= min_gap_frames
            ):
                boundaries.append(frame_idx)
        prev_hist = hist
        frame_idx += 1
    cap.release()

    if frame_idx == 0:
        raise MediaToolError(f"视频一帧都没读到(文件损坏?):{src}")

    # 首尾补齐成完整 shot 区间:0 起始,视频末尾收尾
    edges = [0, *boundaries, frame_idx]
    shots = [
        (round(edges[i] / fps, 3), round(edges[i + 1] / fps, 3))
        for i in range(len(edges) - 1)
        if edges[i] < edges[i + 1]
    ]

    if out_json is not None:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": str(src),
            "fps": fps,
            "n_frames": frame_idx,
            "threshold": threshold,
            "min_gap_s": min_gap_s,
            "n_shots": len(shots),
            "shots": [[s, e] for s, e in shots],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return shots
