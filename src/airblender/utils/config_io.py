"""Standalone YAML loader and bash-style `${VAR:-default}` path resolver."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_SIMPLE_VAR_RE = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
_DEFAULT_VAR_RE = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}")


def expand_env_defaults(path_str: str) -> str:
    """Expand `${VAR}`, `${VAR:-default}`, `$VAR`, and `~` recursively."""
    if not path_str:
        return path_str
    curr = str(path_str)
    prev = None
    while prev != curr:
        prev = curr
        curr = _SIMPLE_VAR_RE.sub(
            lambda m: os.environ[m.group(1)] if os.environ.get(m.group(1), "") != "" else m.group(0),
            curr,
        )
    prev = None
    while prev != curr:
        prev = curr

        def _repl(match: re.Match[str]) -> str:
            val = os.environ.get(match.group(1))
            if val is not None and val != "":
                return val
            return match.group(2) if match.group(2) is not None else ""

        curr = _DEFAULT_VAR_RE.sub(_repl, curr)
    curr = os.path.expandvars(curr)
    curr = os.path.expanduser(curr)
    return curr


def resolve_path(path_val: Any, field_name: str, required: bool = True) -> Path | None:
    """Resolve an optional or required path from config YAML/env vars."""
    if not path_val:
        if required:
            raise ValueError(
                f"Configuration Error: Path for '{field_name}' is not specified. "
                f"Define '{field_name}' in YAML or set the required environment variable."
            )
        return None
    return Path(expand_env_defaults(str(path_val)))


def load_yaml_config(config_path: str | Path, cluster_path: str | Path | None = None) -> dict[str, Any]:
    """Load standalone YAML config, optionally applying cluster base paths if provided."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    if cluster_path:
        cp = Path(cluster_path)
        if cp.exists():
            with open(cp, "r", encoding="utf-8") as cf:
                cluster_cfg = yaml.safe_load(cf) or {}
            cfg = dict(cfg)
            cfg["cluster"] = cluster_cfg
            p_paths = cluster_cfg.get("paths", {})
            for key, env_key in (("scratch_dir", "SCRATCH"), ("data_dir", "DATA_DIR"), ("work_dir", "WORK_DIR")):
                if key in p_paths and env_key not in os.environ:
                    os.environ[env_key] = str(resolve_path(p_paths[key], key, required=False))
    return cfg
