# W3 scope — 人物替换:sprite(人设图白底抠像贴入)与 palette(LAB 色板颜色迁移)双路径 + sprite 宽度时序稳定。
"""人物替换合成器(契约条款 1/3)。

sprite 路径:人设图三视图取「正面」那一格,白底转 alpha(阈值+羽化)得到
RGBA 立绘,按 bbox 高度缩放、脚部对齐 bbox 底边中心贴入;bbox 外像素严格
不动(背景保真度依赖此条,由合成区与 bbox 求交保证)。

palette 路径:提取人设图主色板(kmeans k=8),bbox 区域内向色板做 LAB 空间
颜色迁移(Reinhard 统计量迁移,均值漂移衰减 + 对比度比例钳制防翻白),
再过 edgePreservingFilter 去带状噪点;区域外不动。
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 三视图切分后各视角所在格(0 左 / 1 中 / 2 右)。两套 AI 人设图布局实测一致:
# 左=正面半身特写,中=正面(或 3/4)全身立绘,右=背面 —— sprite 取中格
VIEW_CELL = {"front": 1, "back": 2}

# 白底转 alpha 的距离坡:与纯白欧氏距离 ≤ ALPHA_T0 视为背景(全透明),
# ≥ ALPHA_T1 视为角色(不透明),之间线性过渡 —— 阈值+羽化的一体化实现
ALPHA_T0 = 20.0
ALPHA_T1 = 45.0
ALPHA_SOLID = 128        # 视为「实心内容」的 alpha 下限(内容裁剪用)

# sprite 宽度相对 bbox 宽的最大倍数(契约条款 1:宽超 1.6x 则钳制)
MAX_WIDTH_RATIO = 1.6
# 宽度滑动中值滤波参数(契约条款 3:相邻帧跳变 >15% 时向中值回归防闪烁)
STABILIZE_WINDOW = 5
STABILIZE_JUMP = 0.15

# palette 路径的迁移保护:对比度比例钳制区间 + 均值漂移衰减系数。
# 全量 Reinhard 在动画高反差区域会把暗区整体翻白、几何观感崩,衰减均值漂移
# 让「换色」优先于「换明度」(test_palette_keeps_geometry 的 SSIM 底线也依赖此条)
PALETTE_STD_LO, PALETTE_STD_HI = 0.7, 1.6
PALETTE_MEAN_DAMP = 0.7

# 空帧/退化 bbox 的判定下限(像素):宽或高不足 1px 视为非法
MIN_BOX_SIZE = 1.0


def _load_sheet_cell(sheet_path: str | Path, view: str = "front") -> tuple[np.ndarray, int, int]:
    """读人设图并切出指定视角那一格,返回 (格图像, 格起点 x, 格终点 x)。"""
    if view not in VIEW_CELL:
        raise ValueError(f"未知视角:{view}(支持 {'/'.join(VIEW_CELL)})")
    sheet = cv2.imread(str(sheet_path), cv2.IMREAD_COLOR)
    if sheet is None:
        raise FileNotFoundError(f"人设图读不了(路径错或文件损坏):{Path(sheet_path).resolve()}")
    h, w = sheet.shape[:2]
    cell = VIEW_CELL[view]
    x0, x1 = cell * w // 3, (cell + 1) * w // 3
    return sheet[:, x0:x1], x0, x1


def _white_dist(cell_bgr: np.ndarray) -> np.ndarray:
    """每像素与纯白 (255,255,255) 的欧氏距离(白底越近越趋于 0)。"""
    return np.linalg.norm(255.0 - cell_bgr.astype(np.float32), axis=2)


def extract_sprite(sheet_path: str | Path, view: str = "front") -> tuple[np.ndarray, dict]:
    """人设图 → RGBA sprite。

    白底转 alpha(阈值+羽化:对「与纯白的距离」做线性坡),再裁到实心内容
    的包围盒去掉四周留白;meta 记录切分区域与内容范围(契约条款 1)。
    """
    cell, x0, x1 = _load_sheet_cell(sheet_path, view)
    dist = _white_dist(cell)
    alpha = np.clip((dist - ALPHA_T0) / (ALPHA_T1 - ALPHA_T0), 0.0, 1.0)

    # 只保留格内最大连通域:实测两套 AI 人设图的相邻视图会越界(特写的手臂
    # 伸进正面格 117px),硬切 1/3 会把邻格残条混进 sprite;最大连通域即本
    # 视角主体,顺带滤掉小白点碎屑
    solid = (alpha * 255.0 >= ALPHA_SOLID).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if n <= 1:
        raise ValueError(f"人设图 {sheet_path} 的 {view} 视角格里没抠到内容(全白?)")
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.where(labels == keep)
    alpha = np.where(labels == keep, alpha, 0.0)
    cy1, cy2, cx1, cx2 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1

    sprite = np.dstack([cell[cy1:cy2, cx1:cx2],
                        (alpha[cy1:cy2, cx1:cx2] * 255.0).round().astype(np.uint8)])
    meta = {
        "view": view,
        "cell_index": VIEW_CELL[view],
        "cell_x_range": [x0, x1],
        "content_bbox_in_sheet": [x0 + cx1, cy1, x0 + cx2, cy2],
        "sprite_size": [int(sprite.shape[1]), int(sprite.shape[0])],
        "alpha_thresholds": [ALPHA_T0, ALPHA_T1],
    }
    return sprite, meta


def _valid_bbox(bbox) -> bool:
    """bbox 合法性:4 元素 xyxy 且宽高都 ≥ 1px。"""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return False
    return (x2 - x1) >= MIN_BOX_SIZE and (y2 - y1) >= MIN_BOX_SIZE


def _clip_roi(frame_bgr: np.ndarray, bbox) -> tuple[int, int, int, int] | None:
    """bbox 取整并夹紧到画面内;夹紧后为空返回 None。"""
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(int(round(float(bbox[0]))), w - 1))
    y1 = max(0, min(int(round(float(bbox[1]))), h - 1))
    x2 = max(0, min(int(round(float(bbox[2]))), w))
    y2 = max(0, min(int(round(float(bbox[3]))), h))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return x1, y1, x2, y2


def stabilize_widths(widths: list[float], window: int = STABILIZE_WINDOW,
                     jump: float = STABILIZE_JUMP) -> list[float]:
    """宽度序列的滑动中值滤波(契约条款 3)。

    某帧宽度偏离窗口中值超过 jump 比例(即相邻帧级跳变)时,向中值回归,
    抑制单帧闪烁;缓变趋势(走路推近等)不受影响。窗口取原序列计算,
    修正是单趟的,不迭代扩散。
    """
    src = [float(v) for v in widths]
    out = list(src)
    half = max(1, int(window)) // 2
    for i, v in enumerate(src):
        med = float(np.median(src[max(0, i - half): i + half + 1]))
        if med > 0 and abs(v - med) > jump * med:
            out[i] = med
    return out


class SpriteReplacer:
    """sprite 路径合成器:人设立绘按 bbox 缩放贴入。"""

    def __init__(self, chara_sheet: str | Path, edge_blend_px: int = 12,
                 feather_sigma: float = 8.0) -> None:
        self.sprite, self.meta = extract_sprite(chara_sheet)
        self.edge_blend_px = max(0, int(edge_blend_px))
        self.feather_sigma = float(feather_sigma)

    def target_size(self, bbox) -> tuple[int, int]:
        """按替换规则算 sprite 目标尺寸:按 bbox 高度等比缩放,宽超 1.6x 钳制。

        钳制作用于缩放系数而非只压宽度(保持立绘纵横比,不把角色压瘦)。
        """
        sh, sw = self.sprite.shape[:2]
        bw, bh = float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1])
        scale = min(bh / sh, MAX_WIDTH_RATIO * bw / sw)
        return max(1, round(sw * scale)), max(1, round(sh * scale))

    def replace(self, frame_bgr: np.ndarray, bbox,
                target_w: float | None = None) -> np.ndarray:
        """把 sprite 贴入当前帧,返回新帧(入参不动)。

        target_w 供调用方传入经滑动中值稳定的宽度(契约条款 3);
        不传则按 target_size 现算。空/退化 bbox 拒绝处理,原样返回并记 warning。
        """
        if not _valid_bbox(bbox):
            logger.warning("sprite 路径拒绝空/退化 bbox:%s,本帧原样返回", bbox)
            return frame_bgr
        roi_box = _clip_roi(frame_bgr, bbox)
        if roi_box is None:
            logger.warning("sprite 路径 bbox 夹紧到画面后为空:%s,本帧原样返回", bbox)
            return frame_bgr
        bx1, by1, bx2, by2 = roi_box

        sh, sw = self.sprite.shape[:2]
        if target_w is not None:
            tw = max(1, round(float(target_w)))
            th = max(1, round(tw * sh / sw))
        else:
            tw, th = self.target_size(bbox)

        interp = cv2.INTER_AREA if tw < sw else cv2.INTER_LINEAR
        sprite = cv2.resize(self.sprite, (tw, th), interpolation=interp)
        # 边缘羽化:核对 alpha 做高斯模糊,核宽由 edge_blend_px、sigma 由 feather_sigma
        k = 2 * self.edge_blend_px + 1
        alpha = cv2.GaussianBlur(sprite[:, :, 3], (k, k), self.feather_sigma)

        # 贴入矩形:脚底中心 = bbox 底边中心
        cx = (bx1 + bx2) / 2.0
        px1 = int(round(cx - tw / 2.0))
        py1, py2 = by2 - th, by2

        # 契约:bbox 外像素严格不动 → 合成区 = sprite 矩形 ∩ bbox ∩ 画面
        rx1, ry1 = max(px1, bx1, 0), max(py1, by1, 0)
        rx2, ry2 = min(px1 + tw, bx2, frame_bgr.shape[1]), min(py2, by2, frame_bgr.shape[0])
        if rx2 <= rx1 or ry2 <= ry1:
            logger.warning("sprite 贴入区与 bbox/画面无交集(bbox=%s),本帧原样返回", bbox)
            return frame_bgr
        sx1, sy1 = rx1 - px1, ry1 - py1
        rh, rw = ry2 - ry1, rx2 - rx1
        a = alpha[sy1:sy1 + rh, sx1:sx1 + rw].astype(np.float32) / 255.0
        src = sprite[sy1:sy1 + rh, sx1:sx1 + rw, :3].astype(np.float32)

        out = frame_bgr.copy()
        roi = out[ry1:ry2, rx1:rx2].astype(np.float32)
        out[ry1:ry2, rx1:rx2] = ((1.0 - a[..., None]) * roi
                                 + a[..., None] * src + 0.5).astype(np.uint8)
        return out


class PaletteReplacer:
    """palette 路径合成器:bbox 区域向人设色板做 LAB 颜色迁移。"""

    def __init__(self, chara_sheet: str | Path, k: int = 8) -> None:
        cell, _, _ = _load_sheet_cell(chara_sheet)
        # 只用角色像素提色板:近白背景必须剔除,否则 kmeans 会被背景色吞掉
        fg = cell.reshape(-1, 3)[_white_dist(cell).reshape(-1) > ALPHA_T1]
        if len(fg) < k * 10:
            raise ValueError(f"人设图 {chara_sheet} 前景像素过少({len(fg)}),提不出色板")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _, _, centers = cv2.kmeans(np.ascontiguousarray(fg, dtype=np.float32), k,
                                   None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        self.palette = centers.astype(np.uint8)  # BGR, k 色
        # 色板在 LAB(8bit 缩放尺度)的均值/方差 = 颜色迁移的目标统计量
        lab = cv2.cvtColor(self.palette.reshape(-1, 1, 3),
                           cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        self._tgt_mean = lab.mean(axis=0)
        self._tgt_std = lab.std(axis=0)
        self.meta = {"k": k, "palette_bgr": self.palette.tolist()}

    def replace(self, frame_bgr: np.ndarray, bbox) -> np.ndarray:
        """bbox 区域内向色板做 LAB 颜色迁移 + 边缘保持滤波,区域外不动。

        空/退化 bbox 拒绝处理,原样返回并记 warning(与 sprite 路径一致)。
        """
        if not _valid_bbox(bbox):
            logger.warning("palette 路径拒绝空/退化 bbox:%s,本帧原样返回", bbox)
            return frame_bgr
        roi_box = _clip_roi(frame_bgr, bbox)
        if roi_box is None:
            logger.warning("palette 路径 bbox 夹紧到画面后为空:%s,本帧原样返回", bbox)
            return frame_bgr
        x1, y1, x2, y2 = roi_box

        out = frame_bgr.copy()
        roi = out[y1:y2, x1:x2]
        # 8bit LAB 的 uint8 转换路径保证 L/a/b 都在 0-255 尺度,统计量与色板同尺度
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        for c in range(3):
            src_mean, src_std = float(lab[..., c].mean()), float(lab[..., c].std()) + 1e-6
            ratio = float(np.clip(self._tgt_std[c] / src_std, PALETTE_STD_LO, PALETTE_STD_HI))
            shifted = (lab[..., c] - src_mean) * ratio + self._tgt_mean[c]
            lab[..., c] = lab[..., c] + PALETTE_MEAN_DAMP * (shifted - lab[..., c])
        roi[:] = cv2.cvtColor(np.clip(lab, 0.0, 255.0).astype(np.uint8), cv2.COLOR_LAB2BGR)
        # 去逐像素迁移带来的带状噪点,同时保住边缘(结构)不被抹平
        roi[:] = cv2.edgePreservingFilter(roi, flags=cv2.RECURS_FILTER,
                                          sigma_s=60, sigma_r=0.45)
        return out
