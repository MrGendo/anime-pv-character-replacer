# W1 scope — 配置加载:YAML 读取、env 定位、deep_merge、API key 从环境变量解析。
"""配置加载模块。

返回类型选型:本模块 ``load_config`` 一律返回 ``dict``(不做 dataclass),
理由是配置含嵌套 dict 与 list(如 segments),dict 与 YAML 天然同构,
deep_merge 也直接作用于 dict;引入 dataclass 层只会增加样板代码。
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

# 配置文件定位的回退顺序:显式 path > env ANIME_PV_CONFIG > ./config/pipeline.yaml
CONFIG_ENV_VAR = "ANIME_PV_CONFIG"
DEFAULT_CONFIG_PATH = Path("config/pipeline.yaml")

# 全字段默认值。pipeline.yaml 中的同名键会覆盖这里的值(None 除外),
# 保证新增配置键时旧配置文件仍可加载(向后兼容)。
DEFAULT_CONFIG: dict[str, Any] = {
    "version": "0.1",
    "source": {
        "video_path": "",
        "video_url": "",
        "license_note": "",
    },
    "segments": [],  # list: [{name, start, end, note}]
    "paths": {
        "assets_root": "assets",
        "output_root": "output",
        "workdir_root": "output/work",
    },
    "detection": {
        "method": "yolon",
        "score_thresh": 0.35,
        "device": "cpu",
        "max_persons": 4,
    },
    "replacement": {
        "method": "stylization",
        "api": {
            "endpoint": "",
            "key_env": "ARK_API_KEY",
            "model": "",
            "retries": 3,
            "timeout_s": 30.0,
        },
        "edge_blend_px": 12,
        "feather_sigma": 8.0,
    },
    "tracking": {
        "smoothing_window": 7,
        "max_gap_frames": 8,
    },
    "export": {
        "fps": 23.976,
        "crf": 18,
        "preset": "slow",
    },
    "evaluation": {
        "metrics": ["ssim", "lpips", "clip_i", "temporal_stability"],
        "weights": {
            "ssim": 0.3,
            "lpips": 0.3,
            "clip_i": 0.2,
            "temporal_stability": 0.2,
        },
    },
}


class ConfigError(Exception):
    """配置文件缺失或格式非法。"""


def deep_merge(base: dict, override: dict) -> dict:
    """嵌套合并两个 dict 并返回新 dict(base 不被修改)。

    规则:override 覆盖 base;双方均为 dict 时递归合并;
    override 中值为 None 的键视为「未设置」,保留 base 的值。
    list 不做逐元素合并,整体覆盖(配置里 list 是完整声明,不是增量)。
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path | None = None) -> dict:
    """加载配置,返回合并默认值后的 dict。

    定位顺序:显式 ``path`` > 环境变量 ``ANIME_PV_CONFIG`` > ``./config/pipeline.yaml``。
    显式传入的 path 不存在时直接报错(env/默认路径不存在亦同),
    避免静默回退到默认配置造成「以为改了配置其实没生效」的排障成本。
    """
    if path is not None:
        cfg_path = Path(path)
    elif os.environ.get(CONFIG_ENV_VAR):
        cfg_path = Path(os.environ[CONFIG_ENV_VAR])
    else:
        cfg_path = DEFAULT_CONFIG_PATH

    if not cfg_path.is_file():
        raise ConfigError(
            f"找不到配置文件:{cfg_path}\n"
            f"请检查路径,或设置环境变量 {CONFIG_ENV_VAR} 指向实际配置文件。"
        )

    try:
        with open(cfg_path, encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"配置文件 YAML 解析失败:{cfg_path}\n{e}") from e

    if not isinstance(user_cfg, dict):
        raise ConfigError(f"配置文件顶层必须是映射(dict):{cfg_path}")

    return deep_merge(DEFAULT_CONFIG, user_cfg)


def resolve_api_key(config: dict) -> str:
    """按 replacement.api.key_env 指定的环境变量名读取 API key。

    key 只从环境变量来,任何配置文件/代码里出现明文 key 都违反契约。
    """
    key_env = config["replacement"]["api"].get("key_env", "ARK_API_KEY")
    return os.environ.get(key_env, "")
