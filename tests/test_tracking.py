# W2 scope — tracking 契约测试:相向移动 ID 连续、消失重现 LOST→RECOVER 续 ID、超容忍断轨、平滑无跳变。
"""SimpleTracker 契约测试(合成 detections 序列,不依赖视频)。"""

from __future__ import annotations

from anime_pv_replacer.tracking import SimpleTracker


def box(x: float, y: float = 100.0, w: float = 60.0, h: float = 160.0,
        score: float = 0.9) -> dict:
    """构造一个检测框(x 为左上角 x 坐标)。"""
    return {"bbox": [x, y, x + w, y + h], "score": score}


def test_approaching_boxes_keep_ids() -> None:
    """两框相向移动(不交错):每帧恰两条轨迹,左右目标各持有固定 ID 不互换。"""
    tracker = SimpleTracker(iou_threshold=0.3, max_gap_frames=8, smoothing_window=1)
    left_ids: set[int] = set()
    right_ids: set[int] = set()
    for f in range(20):
        ret = tracker.update([box(100 + 10 * f), box(600 - 10 * f)], f)
        assert len(ret) == 2, f"第 {f} 帧应有 2 条轨迹,实际 {len(ret)}"
        # 按帧内相对位置(而非绝对中点)判左右:两框接近画面中部时绝对判据会失真
        by_x = sorted(ret, key=lambda r: r["bbox"][0])
        left_ids.add(by_x[0]["track_id"])
        right_ids.add(by_x[1]["track_id"])
    assert len(left_ids) == 1 and len(right_ids) == 1, \
        f"左右目标 ID 应各自恒定,实际 left={left_ids} right={right_ids}"
    assert left_ids != right_ids, "两个不同目标不应共用一个 ID"


def test_lost_then_recover_keeps_id() -> None:
    """目标消失 3 帧(< max_gap)后重现:LOST→RECOVER 且续原 ID。"""
    tracker = SimpleTracker(iou_threshold=0.3, max_gap_frames=8, smoothing_window=1)
    tid = tracker.update([box(300)], 0)[0]["track_id"]

    for f in range(1, 10):  # 持续命中,每帧右移 10px
        ret = tracker.update([box(300 + 10 * f)], f)
        assert ret[0]["track_id"] == tid and ret[0]["state"] == "TRACKING"

    for f in range(10, 13):  # 消失 3 帧:轨迹仍返回,状态 LOST,bbox 沿用
        ret = tracker.update([], f)
        assert ret and ret[0]["track_id"] == tid and ret[0]["state"] == "LOST"

    # 重现位置贴着消失前位置(IoU 足以匹配):续原 ID,状态 RECOVER
    ret = tracker.update([box(390)], 13)
    assert ret[0]["track_id"] == tid, "容忍窗口内重现应续原 ID"
    assert ret[0]["state"] == "RECOVER"
    ret = tracker.update([box(400)], 14)  # 再下一帧回到 TRACKING
    assert ret[0]["track_id"] == tid and ret[0]["state"] == "TRACKING"


def test_gap_over_tolerance_breaks_track() -> None:
    """空缺超过 max_gap_frames 后断轨:重现目标拿到新 ID。"""
    tracker = SimpleTracker(iou_threshold=0.3, max_gap_frames=8, smoothing_window=1)
    tid = tracker.update([box(300)], 0)[0]["track_id"]
    for f in range(1, 10):  # 空缺 9 帧 > 8:第 9 帧时轨迹已被归档断轨
        tracker.update([], f)
    ret = tracker.update([box(300)], 10)
    assert ret[0]["track_id"] != tid, "超容忍重现应分配新 ID"
    tracks = tracker.export_tracks()
    assert len(tracks) == 2 and {t["track_id"] for t in tracks} == {tid, ret[0]["track_id"]}


def test_smoothing_damps_bbox_jump() -> None:
    """滑动均值平滑:25px 突跳被压到约 jump/window(≤ 25/5+1),无原始级跳变。

    跳变幅度取 25px:与上一帧框仍保持 IoU≈0.41(> 阈值 0.3)能续上同一轨迹,
    跳得更大(IoU 归零)会按新目标处理,那是匹配器的正确行为而非平滑失效。
    """
    tracker = SimpleTracker(iou_threshold=0.3, max_gap_frames=8, smoothing_window=5)
    xs: list[float] = []
    for f in range(15):
        x = 100.0 if f < 10 else 125.0  # 第 10 帧原始框 x 突跳 25px
        ret = tracker.update([box(x)], f)
        assert len({item["track_id"] for item in ret}) == 1, "IoU 达标的跳变应续原轨迹"
        xs.append(ret[0]["bbox"][0])
    jump = xs[10] - xs[9]
    assert 0 < jump <= 25 / 5 + 1, f"平滑后单帧位移应 ≤ 约 5px,实际 {jump:.1f}"
    max_step = max(abs(xs[i + 1] - xs[i]) for i in range(len(xs) - 1))
    assert max_step <= 25 / 5 + 1, f"全程相邻帧位移均应被平滑压制,最大 {max_step:.1f}"
