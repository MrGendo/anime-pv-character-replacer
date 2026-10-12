# W1 scope — 结构化运行日志:JSONL 落盘 + 同步 print 到 stdout。
"""结构化运行日志。

RunLogger 每条事件写一行 JSON(字段:{ts, run_id, stage, event, data})
到 ``{outdir}/{run_id}.jsonl``,并同步 print 到 stdout,
保证长跑管线 tail 日志文件与终端观察看到的是同一份内容。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class RunLogger:
    """单次 run 的 JSONL 事件日志器。"""

    def __init__(self, run_id: str, outdir: str | Path, stage: str = "") -> None:
        self.run_id = run_id
        self.stage = stage
        self.outdir = Path(outdir)
        # append 模式逐条写入:同一 run 内重复构造 logger 也不会覆盖已有日志
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.outdir / f"{run_id}.jsonl"

    def event(self, event_name: str, **data: Any) -> None:
        """记录一条事件:写 JSONL 文件并 print 到 stdout。

        形参不叫 ``name`` 是刻意的:业务数据里常有 ``name`` 键(如片段名),
        避免与事件名形参撞名;事件名按契约始终用位置参数传。
        data 值若不可 JSON 序列化,退化为 str() 表示,保证日志永不丢行。
        """
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "stage": self.stage,
            "event": event_name,
            "data": data,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)


def get_run_logger(run_id: str, outdir: str | Path, stage: str = "") -> RunLogger:
    """构造 RunLogger 的工厂函数(契约要求的模块入口)。"""
    return RunLogger(run_id=run_id, outdir=outdir, stage=stage)
