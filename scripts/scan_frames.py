#!/usr/bin/env python3
"""Scan PV frames with a VLM to build a content timeline.

Backend selectable via env:
  SCAN_PROVIDER=zhipu (default) or siliconflow
Zhipu glm-4.5v was rate-limited (429) during development, so SiliconFlow
Qwen3-VL-8B (verified working) is used as the fallback scan backend.

Output JSON: [{frame_idx, t_sec, image, raw}]
"""
import base64
import json
import os
import sys
import time
import urllib.request

PROVIDER = os.environ.get("SCAN_PROVIDER", "siliconflow")

if PROVIDER == "zhipu":
    API_KEY = os.environ.get("ZHIPU_API_KEY")
    URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    MODEL = os.environ.get("SCAN_MODEL", "glm-4.5v")
    def make_body(b64):
        return {
            "model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": PROMPT},
            ]}],
            "max_tokens": 4096,
            "temperature": 0.1,
        }
else:
    API_KEY = os.environ.get("SILICONFLOW_API_KEY")
    URL = "https://api.siliconflow.cn/v1/chat/completions"
    MODEL = os.environ.get("SCAN_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    def make_body(b64):
        return {
            "model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": PROMPT},
            ]}],
            "max_tokens": 1500,
            "temperature": 0.1,
        }

PROMPT = """Describe this anime PV frame in under 40 English words. Format strictly:
people: <integer count of anime characters visible, 0 if none>
shot: <one short phrase: e.g. close-up girl / two characters talking / landscape / text / action>
motion: <static | walk | run | action | camera-move>
notes: <hair/clothing colors of each character if visible, else scene note>"""


def encode(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def ask(img_b64, retries=3):
    body = json.dumps(make_body(img_b64)).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    })
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last = None
    for i in range(retries):
        try:
            with opener.open(req, timeout=120) as r:
                data = json.loads(r.read())
                content = data["choices"][0]["message"]["content"]
                # strip <think> blocks if present
                if "</think>" in content:
                    content = content.split("</think>", 1)[1].strip()
                return content
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * (i + 1))
    return f"ERROR: {last}"


def main():
    frames_dir = sys.argv[1]
    out_path = sys.argv[2]
    interval = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    results = []
    for i, fn in enumerate(files):
        p = os.path.join(frames_dir, fn)
        t = (i + 1) * interval  # ffmpeg f_%04d starts at 1
        desc = ask(encode(p))
        entry = {"frame_idx": i + 1, "t_sec": round(t, 1), "image": fn, "raw": desc}
        results.append(entry)
        print(f"[{i+1}/{len(files)}] t={t:.0f}s :: {desc[:160]}", flush=True)
        with open(out_path, "w") as f:
            commit = json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"DONE {len(results)} frames -> {out_path} (provider={PROVIDER} model={MODEL})")


if __name__ == "__main__":
    main()
