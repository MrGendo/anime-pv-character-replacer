# W2 scope — 目标跟踪:帧间 IoU 贪心匹配 + 丢失容忍(max_gap_frames 内重现续 ID)+ bbox 滑动均值平滑。
"""轻量多目标跟踪器。

不引外部跟踪库:匹配只用「上一帧 bbox 与当前检测的 IoU」做贪心配对
(按 IoU 降序取,双方各限配一次),满足 PV 片段内人物数少、运动平缓的场景。
轨迹状态机:TRACKING(本帧命中)→ LOST(本帧未命中但未超容忍)→
RECOVER(LOST 后重新命中,续原 ID);连续空缺超过 max_gap_frames 则断轨,
之后再出现按新目标分配新 ID。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 轨迹状态常量(字符串直接落 JSON,保持产物自描述)
TRACKING = "TRACKING"
LOST = "LOST"
RECOVER = "RECOVER"


def iou(a: list[float], b: list[float]) -> float:
    """两个 xyxy 框的交并比(不相交返回 0)。"""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, iw) * max(0.0, ih)
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


@dataclass
class _Track:
    """单条轨迹的内部状态(raw_boxes 只保留平滑窗口内的原始框)。"""

    track_id: int
    raw_boxes: list[list[float]] = field(default_factory=list)  # 命中帧的原始 bbox,平滑数据源
    log: list[dict] = field(default_factory=list)               # 逐帧输出记录(含 LOST 帧)
    last_frame: int = -1                                        # 最后一次命中检测的帧号
    last_score: float = 0.0
    state: str = TRACKING

    @property
    def frame_start(self) -> int:
        return self.log[0]["frame_idx"]

    @property
    def frame_end(self) -> int:
        return self.log[-1]["frame_idx"]


class SimpleTracker:
    """IoU 匹配 + 丢失容忍 + 滑动均值平滑的跟踪器。"""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_gap_frames: int = 8,
        smoothing_window: int = 7,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_gap_frames = int(max_gap_frames)
        self.smoothing_window = max(1, int(smoothing_window))
        self.active: dict[int, _Track] = {}   # 未断轨的轨迹(track_id → _Track)
        self.finished: list[_Track] = []      # 已断轨轨迹(供导出)
        self._next_id = 1

    def update(self, dets: list[dict], frame_idx: int) -> list[dict]:
        """喂入当前帧检测,返回所有存活轨迹的当前帧记录。

        dets 格式同 PersonDetector.detect:[{"bbox": [x1,y1,x2,y2], "score": float}]。
        返回 [{"track_id", "bbox"(平滑后), "score", "state"}] 按 track_id 升序;
        LOST 轨迹也会返回(bbox 为最后一次平滑值),让上层能观察到空缺区间。
        """
        # 1. 贪心匹配:active 轨迹的上一原始框 vs 当前检测,按 IoU 降序配对
        pairs: list[tuple[float, int, int]] = []
        for tid, track in self.active.items():
            for di, det in enumerate(dets):
                overlap = iou(track.raw_boxes[-1], det["bbox"])
                if overlap >= self.iou_threshold:
                    pairs.append((overlap, tid, di))
        pairs.sort(key=lambda p: p[0], reverse=True)
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for overlap, tid, di in pairs:
            if tid in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(tid)
            matched_dets.add(di)
            self._attach(self.active[tid], dets[di], frame_idx)

        # 2. 未命中轨迹:超容忍断轨归档,否则记 LOST(bbox 沿用上次平滑值,避免跳变)
        for tid in list(self.active):
            if tid in matched_tracks:
                continue
            track = self.active[tid]
            if frame_idx - track.last_frame > self.max_gap_frames:
                self.finished.append(self.active.pop(tid))
                continue
            track.state = LOST
            track.log.append({
                "frame_idx": frame_idx,
                "bbox": track.log[-1]["bbox"],
                "score": round(track.last_score, 4),
                "state": LOST,
            })

        # 3. 未命中检测:新目标开新轨
        for di, det in enumerate(dets):
            if di in matched_dets:
                continue
            track = _Track(track_id=self._next_id, last_frame=frame_idx,
                           last_score=det["score"], state=TRACKING)
            self._attach(track, det, frame_idx)
            self.active[self._next_id] = track
            self._next_id += 1

        return [
            {"track_id": tid, "bbox": track.log[-1]["bbox"],
             "score": track.log[-1]["score"], "state": track.state}
            for tid, track in sorted(self.active.items())
        ]

    def _attach(self, track: _Track, det: dict, frame_idx: int) -> None:
        """命中:记录原始框、输出滑窗均值 bbox,LOST 恢复时状态记 RECOVER。"""
        track.raw_boxes.append(det["bbox"])
        if len(track.raw_boxes) > self.smoothing_window:
            track.raw_boxes.pop(0)
        n = len(track.raw_boxes)
        smoothed = [round(sum(box[i] for box in track.raw_boxes) / n, 1)
                    for i in range(4)]
        track.state = RECOVER if track.state == LOST else TRACKING
        track.last_frame = frame_idx
        track.last_score = float(det["score"])
        track.log.append({
            "frame_idx": frame_idx,
            "bbox": smoothed,
            "score": round(track.last_score, 4),
            "state": track.state,
        })

    def export_tracks(self) -> list[dict]:
        """导出全部轨迹(含已断轨)为可直接 JSON 化的结构。"""
        all_tracks = sorted([*self.finished, *self.active.values()],
                            key=lambda t: t.track_id)
        return [
            {
                "track_id": t.track_id,
                "frame_start": t.frame_start,
                "frame_end": t.frame_end,
                "frames": t.log,
            }
            for t in all_tracks
        ]
