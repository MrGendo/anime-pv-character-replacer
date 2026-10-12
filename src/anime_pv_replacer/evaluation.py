# W4 scope — 自动化评估指标:五维指标(身份/画质/动作/时序/背景)+ 分段线性映射,纯 numpy/opencv 实现。
"""替换质量自动化评估指标(契约条款 1)。

CLIP 不可用(无 GPU 不装 torch),身份维度为降级近似:人设图与替换后人像区
在 LAB 空间做逐通道直方图相关性(cv2.HISTCMP_CORREL)+ 主色板余弦覆盖度,
输出 0-1。全部指标只用 numpy/opencv,不引 torch;输入不合法一律返回 None
并记 warning,绝不抛异常中断整轮(契约硬性要求)。

复用 W3 replacement 模块的内部函数(同包私有复用,避免三视图切分/白底判定
逻辑出现双份漂移):_load_sheet_cell、_white_dist、ALPHA_T1。
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .replacement import ALPHA_T1, _load_sheet_cell, _white_dist

logger = logging.getLogger(__name__)

# ---------- 可调参数(模块级常量,调参只动这里) ----------
HIST_BINS = 32              # LAB 直方图每通道 bin 数
PALETTE_K = 8               # 主色板色数(与 W3 palette 替换路径的 k=8 对齐)
PALETTE_MAX_PIXELS = 20000  # 区域 kmeans 的像素采样上限(全量太慢)
CLARITY_REF_VAR = 1000.0    # Laplacian 方差饱和参考值(达到即清晰度项=1)
EDGE_BAND_PX = 8            # 背景边缘带宽度(EVALUATION_DESIGN:±8px)
FLOW_MIN_PAIRS = 2          # 光流相关至少需要的帧对数(2 点才有相关系数)
FLOW_MIN_SIDE = 16          # 参与光流的最小区域边长 px,再小结果无意义
STATIC_STD_EPS = 1e-3       # 光流幅值序列「视为静止」的标准差阈值
FLICKER_CAP = 100.0         # 原视频静止而替换序列闪烁时的比值封顶(避免 inf 进 JSON)
FRAME_SUFFIXES = {".jpg", ".jpeg", ".png"}

# 分段线性映射节点:0→0分,0.5→60,0.75→80,1.0→100(契约条款 2)
SCORE_KNOTS = ((0.0, 0.0), (0.5, 60.0), (0.75, 80.0), (1.0, 100.0))

# 人设图静态统计缓存:evaluate.py 逐帧调身份指标,不缓存则每帧重做 kmeans
_SHEET_CACHE: dict[str, dict | None] = {}


def _valid_bbox(bbox) -> bool:
    """bbox 合法性:4 元素 xyxy 且宽高都 ≥ 1px(与 replacement 同规则)。"""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return False
    return x2 - x1 >= 1.0 and y2 - y1 >= 1.0


def _clip_roi(shape, bbox) -> tuple[int, int, int, int] | None:
    """bbox 取整并夹紧到画面内;夹紧后为空返回 None。shape 取 (h, w) 或完整帧。"""
    h, w = int(shape[0]), int(shape[1])
    x1 = max(0, min(int(round(float(bbox[0]))), w - 1))
    y1 = max(0, min(int(round(float(bbox[1]))), h - 1))
    x2 = max(0, min(int(round(float(bbox[2]))), w))
    y2 = max(0, min(int(round(float(bbox[3]))), h))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return x1, y1, x2, y2


def _to_gray(img) -> np.ndarray | None:
    """BGR 图转 float64 灰度;非 3 通道 ndarray 返回 None。"""
    if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] < 3 or img.size == 0:
        return None
    return cv2.cvtColor(np.ascontiguousarray(img[:, :, :3]), cv2.COLOR_BGR2GRAY).astype(np.float64)


def _ssim_map(a_gray: np.ndarray, b_gray: np.ndarray) -> np.ndarray:
    """逐像素 SSIM 图(11x11 高斯窗 σ=1.5,标准 C1/C2,0-255 尺度)。"""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    win = (11, 11)
    mu_a = cv2.GaussianBlur(a_gray, win, 1.5)
    mu_b = cv2.GaussianBlur(b_gray, win, 1.5)
    var_a = cv2.GaussianBlur(a_gray * a_gray, win, 1.5) - mu_a ** 2
    var_b = cv2.GaussianBlur(b_gray * b_gray, win, 1.5) - mu_b ** 2
    cov = cv2.GaussianBlur(a_gray * b_gray, win, 1.5) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return num / den


def _read_gray(path: Path) -> np.ndarray | None:
    """读单帧图并转 float64 灰度;失败记 warning 返回 None。"""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("帧图读取失败:%s", path)
        return None
    return _to_gray(img)


def _frame_files(directory) -> dict[int, Path]:
    """目录下逐帧图 → {帧号: 路径}。帧号取文件名数字前缀(W3 产物为 %06d)。"""
    d = Path(directory)
    if not d.is_dir():
        logger.warning("帧目录不存在:%s", d)
        return {}
    out: dict[int, Path] = {}
    for p in sorted(d.iterdir()):
        if p.suffix.lower() not in FRAME_SUFFIXES or not p.stem.isdigit():
            continue
        out[int(p.stem)] = p
    if not out:
        logger.warning("帧目录里没有可用的逐帧图:%s", d)
    return out


def _lab_palette(lab_pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LAB 像素 → kmeans 主色板,返回 (色心 (k,3), 权重 (k,))。

    超过 PALETTE_MAX_PIXELS 时按等距步长采样,kmeans 对采样量不敏感。
    """
    px = np.ascontiguousarray(lab_pixels.reshape(-1, 3), dtype=np.float32)
    if len(px) > PALETTE_MAX_PIXELS:
        px = px[:: int(np.ceil(len(px) / PALETTE_MAX_PIXELS))]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(px, PALETTE_K, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=PALETTE_K).astype(np.float32)
    return centers, counts / counts.sum()


def _palette_coverage(ref_centers: np.ndarray, ref_weights: np.ndarray,
                      qry_centers: np.ndarray) -> float:
    """色板覆盖度:ref 每主色与 qry 各主色方向余弦的最大值,按 ref 权重加权。

    LAB 值先减 128 中心化再取方向,让「色相/明暗方向」而非绝对亮度主导相似度。
    """
    u = ref_centers.astype(np.float64) - 128.0
    v = qry_centers.astype(np.float64) - 128.0
    u = u / np.clip(np.linalg.norm(u, axis=1, keepdims=True), 1e-6, None)
    v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-6, None)
    cos = np.clip(u @ v.T, 0.0, 1.0)  # 方向相反视为完全不匹配(截 0)
    return float((cos.max(axis=1) * ref_weights).sum())


def _sheet_stats(sheet_path) -> dict | None:
    """人设图静态统计(前景 LAB 直方图 + 主色板),按路径缓存。"""
    key = str(Path(sheet_path).resolve())
    if key not in _SHEET_CACHE:
        try:
            cell, _, _ = _load_sheet_cell(sheet_path)
        except (FileNotFoundError, ValueError) as e:
            logger.warning("人设图加载失败(%s):%s", sheet_path, e)
            _SHEET_CACHE[key] = None
            return None
        fg = _white_dist(cell) > ALPHA_T1
        if int(fg.sum()) < PALETTE_K * 10:
            logger.warning("人设图前景像素过少,提不出色板:%s", sheet_path)
            _SHEET_CACHE[key] = None
            return None
        lab = cv2.cvtColor(cell, cv2.COLOR_BGR2LAB)
        fg_px = np.ascontiguousarray(lab[fg]).reshape(-1, 1, 3)
        hists = [cv2.calcHist([fg_px], [c], None, [HIST_BINS], [0.0, 256.0])
                 for c in range(3)]
        centers, weights = _lab_palette(lab[fg])
        _SHEET_CACHE[key] = {"hists": hists, "centers": centers, "weights": weights}
    return _SHEET_CACHE[key]


def clip_identity_score(orig_frame, replaced_frame, sheet_path, bbox) -> float | None:
    """人设图 vs 替换后人像区的身份一致性(0-1)。

    **CLIP 不可用时的降级近似**(契约条款 1):LAB 空间逐通道直方图相关性
    (cv2.HISTCMP_CORREL,负相关截 0)+ 主色板余弦覆盖度,0.6/0.4 加权。
    orig_frame 仅做形状校验(与替换帧同帧同尺寸),分数只看替换后区域 —— 与
    CLIP image-image 语义对齐(人设图 vs 替换结果),不引入原图对比项。
    """
    if not isinstance(replaced_frame, np.ndarray) or replaced_frame.ndim != 3 or replaced_frame.size == 0:
        logger.warning("clip_identity_score 收到非法替换帧,返回 None")
        return None
    if isinstance(orig_frame, np.ndarray) and orig_frame.shape[:2] != replaced_frame.shape[:2]:
        logger.warning("clip_identity_score 原始帧与替换帧尺寸不一致,返回 None")
        return None
    stats = _sheet_stats(sheet_path)
    if stats is None:
        return None
    if not _valid_bbox(bbox):
        logger.warning("clip_identity_score 收到非法 bbox:%s", bbox)
        return None
    roi = _clip_roi(replaced_frame.shape, bbox)
    if roi is None:
        logger.warning("clip_identity_score bbox 夹紧后为空:%s", bbox)
        return None
    x1, y1, x2, y2 = roi
    region_lab = cv2.cvtColor(replaced_frame[y1:y2, x1:x2], cv2.COLOR_BGR2LAB)
    region_px = np.ascontiguousarray(region_lab.reshape(-1, 1, 3))

    corr = []
    for c in range(3):
        h = cv2.calcHist([region_px], [c], None, [HIST_BINS], [0.0, 256.0])
        corr.append(float(cv2.compareHist(stats["hists"][c], h, cv2.HISTCMP_CORREL)))
    hist_term = float(np.clip(np.mean(corr), 0.0, 1.0))

    centers, weights = _lab_palette(region_lab)
    palette_term = _palette_coverage(stats["centers"], stats["weights"], centers)
    return round(0.6 * hist_term + 0.4 * palette_term, 4)


def quality_score(frame_bgr, ref_bgr=None) -> float | None:
    """替换后区域画质(0-1):Laplacian 方差(清晰度)+ 替换前后差(结构保留)。

    ref_bgr 缺省时只算清晰度项(单帧也能用);给了 ref 则 0.5 清晰度 +
    0.5 SSIM(结构保留)。ref 形状非法时记 warning 退化为仅清晰度项。
    """
    gray = _to_gray(frame_bgr)
    if gray is None:
        logger.warning("quality_score 收到非法帧,返回 None")
        return None
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var())
    clarity = min(1.0, (max(lap_var, 0.0) / CLARITY_REF_VAR) ** 0.5)
    if ref_bgr is None:
        return round(clarity, 4)
    ref_gray = _to_gray(ref_bgr)
    if ref_gray is None or ref_gray.shape != gray.shape:
        logger.warning("quality_score 的 ref 帧非法或尺寸不一致,退化为仅清晰度项")
        return round(clarity, 4)
    structure = float(_ssim_map(gray, ref_gray).mean())
    return round(0.5 * clarity + 0.5 * structure, 4)


def _flow_mag(g0: np.ndarray, g1: np.ndarray) -> float:
    """Farneback 光流平均幅值(px/帧),入参为同尺寸 float 灰度。"""
    flow = cv2.calcOpticalFlowFarneback(
        g0.astype(np.float32), g1.astype(np.float32), None,
        0.5, 3, 15, 3, 5, 1.2, 0)
    return float(np.hypot(flow[..., 0], flow[..., 1]).mean())


def _aligned_pairs(orig_frames_dir, replaced_frames_dir, bbox_per_frame):
    """两个帧目录 + 逐帧 bbox → 可计算的相邻帧对迭代器。

    只取「两目录同名、帧号连续、两帧都有合法 bbox」的对;任一帧读不出记
    warning 跳过。yield (帧号a, 原始灰度a/b, 替换灰度a/b, 区域坐标)。
    """
    mo, mr = _frame_files(orig_frames_dir), _frame_files(replaced_frames_dir)
    boxes = {int(k): v for k, v in (bbox_per_frame or {}).items() if _valid_bbox(v)}
    common = sorted(set(mo) & set(mr))
    for a, b in zip(common, common[1:]):
        if b - a != 1 or a not in boxes or b not in boxes:
            continue
        ga, gb = _read_gray(mo[a]), _read_gray(mo[b])
        ra, rb = _read_gray(mr[a]), _read_gray(mr[b])
        if ga is None or gb is None or ra is None or rb is None:
            continue
        if ga.shape != gb.shape or ra.shape != rb.shape or ga.shape != ra.shape:
            logger.warning("帧 %d/%d 尺寸不一致,跳过该帧对", a, b)
            continue
        yield a, ga, gb, ra, rb, boxes[a]


def motion_consistency(orig_frames_dir, replaced_frames_dir, bbox_per_frame) -> float | None:
    """替换前后 bbox 区域光流幅值序列的 Pearson 相关(0-1)。

    每个相邻帧对在出发帧的 bbox 区域内算 Farneback 平均幅值,得到两条序列
    后取相关系数:动作节奏保留 → 高分;一方静止另一方运动 → 0;双方都静止
    (序列无方差,相关系数无定义)→ 按完全一致返回 1.0。
    """
    seq_o: list[float] = []
    seq_r: list[float] = []
    for _, ga, gb, ra, rb, bbox in _aligned_pairs(orig_frames_dir, replaced_frames_dir, bbox_per_frame):
        roi = _clip_roi(ga.shape, bbox)
        if roi is None:
            continue
        x1, y1, x2, y2 = roi
        if x2 - x1 < FLOW_MIN_SIDE or y2 - y1 < FLOW_MIN_SIDE:
            continue
        seq_o.append(_flow_mag(ga[y1:y2, x1:x2], gb[y1:y2, x1:x2]))
        seq_r.append(_flow_mag(ra[y1:y2, x1:x2], rb[y1:y2, x1:x2]))
    if len(seq_o) < FLOW_MIN_PAIRS:
        logger.warning("motion_consistency 有效帧对不足 %d 对(实际 %d),返回 None",
                       FLOW_MIN_PAIRS, len(seq_o))
        return None
    so, sr = np.asarray(seq_o), np.asarray(seq_r)
    if so.std() < STATIC_STD_EPS and sr.std() < STATIC_STD_EPS:
        return 1.0
    if so.std() < STATIC_STD_EPS or sr.std() < STATIC_STD_EPS:
        return 0.0
    corr = float(np.corrcoef(so, sr)[0, 1])
    if np.isnan(corr):
        logger.warning("motion_consistency 相关系数为 NaN,返回 None")
        return None
    return round(float(np.clip(corr, 0.0, 1.0)), 4)


def temporal_flicker(orig_frames_dir, replaced_frames_dir, bbox_per_frame) -> float | None:
    """闪烁比:替换区相邻帧均值差序列 ÷ 原视频同序列(契约条款 1)。

    1.0 = 与原视频一致;<1 比原视频更平滑;>1 闪烁。区域取相邻两帧 bbox 的
    交集(只比较同被跟踪的位置)。原视频完全静止而替换序列有变化时,比值
    数学上无界,封顶 FLICKER_CAP 并记 warning。
    """
    # 键统一转 int:帧号在调用方可能是 str(json 反序列化)或 int
    boxes = {int(k): v for k, v in (bbox_per_frame or {}).items()}
    sum_o = sum_r = 0.0
    n_pairs = 0
    for a, ga, gb, ra, rb, bbox in _aligned_pairs(orig_frames_dir, replaced_frames_dir, boxes):
        roi_a = _clip_roi(ga.shape, bbox)
        roi_b = _clip_roi(gb.shape, boxes[a + 1])
        if roi_a is None or roi_b is None:
            continue
        x1, y1 = max(roi_a[0], roi_b[0]), max(roi_a[1], roi_b[1])
        x2, y2 = min(roi_a[2], roi_b[2]), min(roi_a[3], roi_b[3])
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        sum_o += float(np.abs(gb[y1:y2, x1:x2] - ga[y1:y2, x1:x2]).mean())
        sum_r += float(np.abs(rb[y1:y2, x1:x2] - ra[y1:y2, x1:x2]).mean())
        n_pairs += 1
    if n_pairs < 1:
        logger.warning("temporal_flicker 没有任何有效帧对(实际 %d),返回 None", n_pairs)
        return None
    if sum_o < 1e-6:
        if sum_r < 1e-6:
            return 1.0
        logger.warning("原视频替换区静止而替换序列在变,闪烁比封顶 %.1f", FLICKER_CAP)
        return FLICKER_CAP
    return round(sum_r / sum_o, 4)


def background_fidelity(orig, replaced, bbox=None) -> float | None:
    """背景保真度(0-1):bbox 外逐像素相等率 + 边缘带 SSIM,各 0.5 权重。

    bbox 缺省时退化为全图逐像素相等率(无「替换区」则边缘带无意义)。
    逐像素相等 = BGR 三通道全部相等,应严格为 1.0(工程强制项)。
    """
    if (not isinstance(orig, np.ndarray) or not isinstance(replaced, np.ndarray)
            or orig.ndim != 3 or replaced.ndim != 3 or orig.size == 0
            or orig.shape != replaced.shape):
        logger.warning("background_fidelity 收到非法帧对,返回 None")
        return None
    h, w = orig.shape[:2]
    if bbox is None:
        return round(float((orig == replaced).all(axis=2).mean()), 4)
    if not _valid_bbox(bbox):
        logger.warning("background_fidelity 收到非法 bbox:%s", bbox)
        return None
    roi = _clip_roi((h, w), bbox)
    if roi is None:
        logger.warning("background_fidelity bbox 夹紧后为空:%s", bbox)
        return None
    x1, y1, x2, y2 = roi
    outside = np.ones((h, w), dtype=bool)
    outside[y1:y2, x1:x2] = False
    equal_rate = float((orig == replaced).all(axis=2)[outside].mean())

    # 边缘带:bbox 外扩 EDGE_BAND_PX 的环(仍在 bbox 外),查合成接缝是否伤背景
    band = np.zeros((h, w), dtype=bool)
    band[max(0, y1 - EDGE_BAND_PX):min(h, y2 + EDGE_BAND_PX),
         max(0, x1 - EDGE_BAND_PX):min(w, x2 + EDGE_BAND_PX)] = True
    band &= outside
    if int(band.sum()) < 1:  # bbox 占满全图 → 没有背景可评,只看相等率
        return round(equal_rate, 4)
    ga, gr = _to_gray(orig), _to_gray(replaced)
    edge_ssim = float(_ssim_map(ga, gr)[band].mean())
    return round(0.5 * equal_rate + 0.5 * edge_ssim, 4)


def piecewise_score(x) -> float:
    """分段线性映射到 0-100 分:0→0 / 0.5→60 / 0.75→80 / 1.0→100(契约条款 2)。"""
    x = min(1.0, max(0.0, float(x)))
    for (x0, y0), (x1, y1) in zip(SCORE_KNOTS, SCORE_KNOTS[1:]):
        if x <= x1:
            t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
            return round(y0 + t * (y1 - y0), 2)
    return 100.0


def flicker_score(ratio) -> float | None:
    """闪烁比 → 得分:偏离 1 越远越差,经分段线性反向映射(1.0→100 分)。"""
    if ratio is None:
        return None
    return round(100.0 - piecewise_score(abs(1.0 - float(ratio))), 2)
