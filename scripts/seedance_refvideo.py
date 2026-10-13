#!/usr/bin/env python3
"""seedance_refvideo.py — reference_video mode: video-to-video character replacement.

Discovered via API error probing (docs site was bot-walled):
  - Seedance 2.5 accepts {"type":"video_url","video_url":{"url":<WEB URL>},"role":"reference_video"}
  - reference_video MUST be a web URL (base64 rejected with InvalidParameter)
So the source segment is uploaded to a public URL (GitHub raw on a probe branch),
then submitted together with the character sheet as reference_image.

Usage: seedance_refvideo.py <video_web_url> <sheet_path> <out_mp4> [duration] [resolution]
Env: ARK_API_KEY
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
    "Reference video 1 is the SOURCE footage. Reference image 2 is a character "
    "design sheet. Re-render the source video exactly: same camera, same framing, "
    "same background, same character motion and timing, but REPLACE the male "
    "character's appearance with the design in image 2 (same hairstyle, hair "
    "color, outfit). Japanese TV anime cel-shaded style. No text, no watermark."
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
            with opener.open(req, timeout=180) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"ARK request failed: {last}")


def _b64(path):
    data = open(path, "rb").read()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def submit(video_url, sheet, prompt, resolution, duration, ratio="16:9"):
    content = [
        {"type": "video_url", "video_url": {"url": video_url}, "role": "reference_video"},
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
    video_url, sheet, out_mp4 = sys.argv[1], sys.argv[2], sys.argv[3]
    duration = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    resolution = sys.argv[5] if len(sys.argv) > 5 else "720p"
    t0 = time.time()
    task_id = submit(video_url, sheet, PROMPT_TEMPLATE, resolution, duration)
    print(f"task {task_id} submitted (reference_video mode, dur={duration}s)", flush=True)
    d = {}
    status = ""
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
            "resolution": resolution, "model": MODEL, "mode": "reference_video",
            "video_url": video_url, "sheet": sheet, "out": out_mp4,
            "size_mb": round(len(data) / 1e6, 2)}
    json.dump(meta, open(out_mp4.replace(".mp4", "_meta.json"), "w"), indent=1)
    print("DONE " + json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
