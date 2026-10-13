#!/usr/bin/env python3
"""seedance_replace.py — Seedance 2.5 (video generation model) character replacement path.

Unlike sprite/palette (local compositing), this path asks a video generation
model to REGENERATE the clip with the character design applied, using:
  ref image 1: character design sheet (front view crop)
  ref image 2: first frame of the original segment (scene/composition anchor)
  prompt: replace characters with the reference design, keep camera/background

API contract mirrors the production-tested ark.py client:
  POST /api/v3/contents/generations/tasks  (multi-image => role=reference_image)
  poll GET /tasks/{id} until succeeded/failed, download video_url.

Requires env ARK_API_KEY. Usage tokens recorded for cost transparency.
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
    "Japanese TV anime PV scene. Regenerate this exact scene, but REPLACE the "
    "human characters with the character shown in the first reference image "
    "(character design sheet): keep that character's hairstyle, hair color, eye "
    "color and outfit exactly as designed. Keep the camera framing, background "
    "art, lighting and color grading of the second reference image (video first "
    "frame) unchanged. Natural anime motion, cel-shaded, high quality, no text, "
    "no watermark."
)


def _hdr():
    if not API_KEY:
        print("[FAILED] ARK_API_KEY 未设置 — Seedance 为付费路径，需要显式提供 key 才能运行", file=sys.stderr)
        sys.exit(2)
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def _post(url, payload):
    # 火山方舟国内端点：优先系统代理，失败自动绕代理直连一次
    body = json.dumps(payload).encode()
    attempts = []
    sp = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or ""
    if sp:
        attempts.append({"http": sp, "https": sp})
    attempts.append({})  # 直连兜底
    last = None
    for proxies in attempts:
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
            req = urllib.request.Request(url, data=body, headers=_hdr())
            with opener.open(req, timeout=120) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            last = e
            continue
    raise RuntimeError(f"ARK submit failed after proxy attempts: {last}")


def _get(url):
    req = urllib.request.Request(url, headers=_hdr())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=120) as r:
        return json.loads(r.read())


def _b64(path):
    data = open(path, "rb").read()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def submit(sheet_path, first_frame_path, prompt, resolution, duration, ratio="16:9"):
    content = [
        {"type": "image_url", "image_url": {"url": _b64(sheet_path)}, "role": "reference_image"},
        {"type": "image_url", "image_url": {"url": _b64(first_frame_path)}, "role": "reference_image"},
        {"type": "text", "text": prompt},
    ]
    payload = {"model": MODEL, "content": content,
               "generate": {"resolution": resolution, "duration": duration, "ratio": ratio}}
    d = _post(f"{ARK_BASE}/contents/generations/tasks", payload)
    if "id" not in d:
        raise RuntimeError(f"no task id: {json.dumps(d, ensure_ascii=False)[:300]}")
    return d["id"]


def wait(task_id):
    for i in range(POLL_MAX):
        time.sleep(POLL_INTERVAL)
        d = _get(f"{ARK_BASE}/contents/generations/tasks/{task_id}")
        status = d.get("status")
        if status in ("succeeded", "failed"):
            return d
        if i % 5 == 0:
            print(f"  ...poll {i}: {status}", flush=True)
    return {"status": "timeout"}


def download(url, dest):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=300) as r:
        data = r.read()
    open(dest, "wb").write(data)
    return dest


def main():
    # argv: sheet first_frame out_mp4 [duration] [resolution]
    sheet, first_frame, out_mp4 = sys.argv[1], sys.argv[2], sys.argv[3]
    duration = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    resolution = sys.argv[5] if len(sys.argv) > 5 else "720p"
    t0 = time.time()
    task_id = submit(sheet, first_frame, PROMPT_TEMPLATE, resolution, duration)
    print(f"task {task_id} submitted (dur={duration}s res={resolution})", flush=True)
    d = wait(task_id)
    status = d.get("status")
    usage = (d.get("usage") or {}).get("completion_tokens", 0)
    if status != "succeeded":
        print(json.dumps({"status": status, "error": d.get("error"), "task_id": task_id}, ensure_ascii=False))
        sys.exit(1)
    url = (d.get("content") or {}).get("video_url", "")
    download(url, out_mp4)
    meta = {"task_id": task_id, "status": status, "usage_completion_tokens": usage,
            "wall_s": round(time.time() - t0, 1), "duration_s": duration,
            "resolution": resolution, "model": MODEL, "out": out_mp4,
            "size_mb": round(os.path.getsize(out_mp4) / 1e6, 2)}
    meta_path = out_mp4.replace(".mp4", "_meta.json")
    json.dump(meta, open(meta_path, "w"), indent=1)
    print("DONE " + json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
