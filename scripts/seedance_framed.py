#!/usr/bin/env python3
"""seedance_framed.py — first/last-frame anchored Seedance generation.

Motion-preserving replacement: instead of a single first frame + prompt text,
we drive each sub-shot with its REAL first and last frames from the original
PV (motion trajectory locked by the endpoints) plus the character design
sheet as reference. The model interpolates the same action with the new
character design.

Usage:
  seedance_framed.py <sheet> <first_frame> <last_frame> <out_mp4> [duration] [resolution]

Requires env ARK_API_KEY. Usage tokens recorded in <out>_meta.json.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.request

ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"
MODEL = os.environ.get("ARK_MODEL", "doubao-seedance-2-5-260628")
API_KEY = os.environ.get("ARK_API_KEY", "")
POLL_INTERVAL = 8
POLL_MAX = 110

PROMPT_TEMPLATE = (
    "Frame-interpolation anime task. The FIRST frame and LAST frame of this "
    "clip are given as reference images 1 and 2: animate naturally BETWEEN "
    "them, keeping the exact same camera, background, scene composition and "
    "character motion as these two keyframes. Reference image 3 is a "
    "character design sheet: REPLACE the main character's appearance with "
    "this design (same hairstyle, hair color, outfit), keeping the pose and "
    "motion of the keyframes unchanged. Japanese TV anime style, cel-shaded, "
    "no text, no watermark."
)


def _hdr():
    if not API_KEY:
        print("[FAILED] ARK_API_KEY 未设置", file=sys.stderr)
        sys.exit(2)
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def _open(url, data=None):
    body = json.dumps(data).encode() if data is not None else None
    attempts = []
    sp = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or ""
    if sp:
        attempts.append({"http": sp, "https": sp})
    attempts.append({})
    last = None
    for proxies in attempts:
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
            req = urllib.request.Request(url, data=body, headers=_hdr())
            with opener.open(req, timeout=120) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"ARK request failed: {last}")


def _b64(path):
    data = open(path, "rb").read()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def submit(sheet, first, last, prompt, resolution, duration, ratio="16:9"):
    content = [
        {"type": "image_url", "image_url": {"url": _b64(first)}, "role": "reference_image"},
        {"type": "image_url", "image_url": {"url": _b64(last)}, "role": "reference_image"},
        {"type": "image_url", "image_url": {"url": _b64(sheet)}, "role": "reference_image"},
        {"type": "text", "text": prompt},
    ]
    payload = {"model": MODEL, "content": content,
               "generate": {"resolution": resolution, "duration": duration, "ratio": ratio}}
    d = _open(f"{ARK_BASE}/contents/generations/tasks", payload)
    if "id" not in d:
        raise RuntimeError(f"no task id: {json.dumps(d, ensure_ascii=False)[:300]}")
    return d["id"]


def main():
    sheet, first, last, out_mp4 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    duration = int(sys.argv[4 + 1]) if len(sys.argv) > 5 else 5
    resolution = sys.argv[6] if len(sys.argv) > 6 else "720p"
    t0 = time.time()
    task_id = submit(sheet, first, last, PROMPT_TEMPLATE, resolution, duration)
    print(f"task {task_id} submitted (dur={duration}s res={resolution})", flush=True)
    status = ""
    d = {}
    for i in range(POLL_MAX):
        time.sleep(POLL_INTERVAL)
        d = _open(f"{ARK_BASE}/contents/generations/tasks/{task_id}")
        status = d.get("status")
        if status in ("succeeded", "failed"):
            break
        if i % 5 == 0:
            print(f"  ...poll {i}: {status}", flush=True)
    usage = (d.get("usage") or {}).get("completion_tokens", 0)
    if status != "succeeded":
        print(json.dumps({"status": status, "error": d.get("error"), "task_id": task_id}, ensure_ascii=False))
        sys.exit(1)
    url = (d.get("content") or {}).get("video_url", "")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=300) as r:
        data = r.read()
    open(out_mp4, "wb").write(data)
    meta = {"task_id": task_id, "status": status, "usage_completion_tokens": usage,
            "wall_s": round(time.time() - t0, 1), "duration_s": duration,
            "resolution": resolution, "model": MODEL, "mode": "first_last_frame",
            "refs": [sheet, first, last], "out": out_mp4,
            "size_mb": round(len(data) / 1e6, 2)}
    json.dump(meta, open(out_mp4.replace(".mp4", "_meta.json"), "w"), indent=1)
    print("DONE " + json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
