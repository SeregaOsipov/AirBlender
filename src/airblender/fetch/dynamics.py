"""Ordered meteorological dynamics resolver with pre-flight probing and automatic license fallback."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from airblender.fetch.ecmwf_ifs import download_ifs_forecast_mars, probe_ifs_mars_access
from airblender.fetch.opendata import (
    S3_OPENDATA_BASE_URL,
    download_opendata_0p1_sfc_daily,
    probe_opendata_0p1,
)


SUPPORTED_PROVIDERS = ("ifs_mars_9km", "ifs_0p1_sfc_cams_ml", "cams_0p4_ml")
_MARS_PROBE_CACHE: dict[tuple[str, str, str], tuple[bool, str]] = {}


@dataclass
class DynamicsResolution:
    """Resolved meteorological boundary/initial provider and file locations."""

    provider: str
    met_sfc_dir: Path
    met_sfc_prefix: str
    met_ml_dir: Path
    met_ml_prefix: str
    sfc_res_deg: float
    ml_res_deg: float
    fallback_reasons: dict[str, str] = field(default_factory=dict)
    attempts: list[dict[str, str]] = field(default_factory=list)
    resolved_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["met_sfc_dir"] = str(self.met_sfc_dir)
        d["met_ml_dir"] = str(self.met_ml_dir)
        return d


def _normalize_grid_str(raw_grid: Any) -> str:
    if isinstance(raw_grid, (list, tuple)) and len(raw_grid) == 2:
        return f"{float(raw_grid[0]):g}/{float(raw_grid[1]):g}"
    if raw_grid:
        return str(raw_grid)
    return "0.08/0.08"


def _write_provenance(provenance_dir: Path, res: DynamicsResolution) -> None:
    provenance_dir.mkdir(parents=True, exist_ok=True)
    out_fp = provenance_dir / "dynamics_provenance.json"
    tmp_fp = out_fp.with_suffix(f".json.tmp.{os.getpid()}")
    tmp_fp.write_text(json.dumps(res.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_fp, out_fp)


def resolve_and_fetch_dynamics(
    cycle_date: str,
    *,
    cams_dir: str | Path,
    out_dir: str | Path | None = None,
    dynamics_cfg: Mapping[str, Any] | None = None,
    leadtimes_hours: Sequence[int] | None = None,
    area: Sequence[float] | None = None,
    time_str: str = "00:00",
    provenance_dir: str | Path | None = None,
    force: bool = False,
) -> DynamicsResolution:
    """
    Resolve the highest-resolution usable meteorological dynamics provider following:
      1. `DYNAMICS_PROVIDER` environment override (if set and != "auto")
      2. `dynamics_cfg` (`primary` -> `fallback_chain` when `mode == "auto"`)
      3. Safe fallback to `cams_0p4_ml` on missing/expired MARS credentials or valid-time blocks.
    """
    cams_path = Path(cams_dir)
    target_dir = Path(out_dir) if out_dir is not None else (cams_path.parent / "IFS")
    cfg = dict(dynamics_cfg or {})
    env_override = os.environ.get("DYNAMICS_PROVIDER", "").strip()

    if not cfg and not env_override:
        res = DynamicsResolution(
            provider="cams_0p4_ml",
            met_sfc_dir=cams_path,
            met_sfc_prefix="ECMWF-CAMS_sfc",
            met_ml_dir=cams_path,
            met_ml_prefix="ECMWF-CAMS_ml",
            sfc_res_deg=0.4,
            ml_res_deg=0.4,
            resolved_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        if provenance_dir is not None:
            _write_provenance(Path(provenance_dir), res)
        return res

    mode = str(cfg.get("mode", "auto")).strip().lower()
    if env_override and env_override.lower() != "auto":
        candidates = [env_override]
        strict_mode = True
    else:
        primary = str(cfg.get("primary", "ifs_mars_9km")).strip()
        fallback_chain = [str(p).strip() for p in (cfg.get("fallback_chain") or ["ifs_0p1_sfc_cams_ml", "cams_0p4_ml"])]
        if mode == "pinned":
            candidates = [primary]
            strict_mode = True
        else:
            candidates = []
            for p in [primary, *fallback_chain, "cams_0p4_ml"]:
                if p and p not in candidates:
                    candidates.append(p)
            strict_mode = False

    steps = sorted({int(h) for h in (leadtimes_hours if leadtimes_hours is not None else range(0, 121, 3))})
    max_step_h = max(steps) if steps else 0
    mars_cfg = dict(cfg.get("mars") or {})
    od_cfg = dict(cfg.get("opendata") or {})
    mars_grid = _normalize_grid_str(mars_cfg.get("grid", "0.08/0.08"))
    mars_rc = mars_cfg.get("rc_file")
    mars_probe_timeout = float(mars_cfg.get("probe_timeout_sec", 45.0))
    s3_base = str(od_cfg.get("s3_base_url", S3_OPENDATA_BASE_URL))

    fallback_reasons: dict[str, str] = {}
    attempts: list[dict[str, str]] = []

    for candidate in candidates:
        if candidate not in SUPPORTED_PROVIDERS:
            reason = f"Unsupported dynamics provider {candidate!r}"
            fallback_reasons[candidate] = reason
            attempts.append({"provider": candidate, "status": "unsupported", "reason": reason})
            continue

        if candidate == "ifs_mars_9km":
            try:
                mars_type = str(mars_cfg.get("type", "fc"))
                cache_key = (cycle_date.replace("-", "")[:8], mars_type.lower(), mars_grid)
                cached_probe = None if force else _MARS_PROBE_CACHE.get(cache_key)
                if cached_probe is not None:
                    ok, reason = cached_probe
                    if not ok:
                        fallback_reasons[candidate] = reason
                        attempts.append({"provider": candidate, "status": "unavailable", "reason": reason})
                        if strict_mode:
                            raise RuntimeError(f"Pinned dynamics provider '{candidate}' unavailable: {reason}")
                        print(f"[Dynamics-Fallback] Reusing cached MARS probe result ({reason}); checking next fallback...", flush=True)
                        continue
                else:
                    ok, reason = probe_ifs_mars_access(
                        cycle_date,
                        ifs_type=mars_type,
                        time_str=time_str,
                        max_step_h=max_step_h,
                        area=area,
                        grid=mars_grid,
                        rc_file=mars_rc,
                        timeout_sec=mars_probe_timeout,
                    )
                    if not ok:
                        _MARS_PROBE_CACHE[cache_key] = (False, reason)
                if not ok:
                    fallback_reasons[candidate] = reason
                    attempts.append({"provider": candidate, "status": "unavailable", "reason": reason})
                    if strict_mode:
                        raise RuntimeError(f"Pinned dynamics provider '{candidate}' unavailable: {reason}")
                    print(f"[Dynamics-Fallback] Primary '{candidate}' unavailable ({reason}); checking next fallback...", flush=True)
                    continue

                download_ifs_forecast_mars(
                    cycle_date,
                    target_dir,
                    ifs_type=str(mars_cfg.get("type", "fc")),
                    time_str=time_str,
                    leadtimes_hours=steps,
                    area=area,
                    grid=mars_grid,
                    force=force,
                )
                attempts.append({"provider": candidate, "status": "selected", "reason": "ok"})
                res = DynamicsResolution(
                    provider="ifs_mars_9km",
                    met_sfc_dir=target_dir,
                    met_sfc_prefix="ECMWF-IFS_sfc",
                    met_ml_dir=target_dir,
                    met_ml_prefix="ECMWF-IFS_ml",
                    sfc_res_deg=float(mars_grid.split("/")[0]) if "/" in mars_grid else 0.08,
                    ml_res_deg=float(mars_grid.split("/")[0]) if "/" in mars_grid else 0.08,
                    fallback_reasons=fallback_reasons,
                    attempts=attempts,
                    resolved_at_utc=datetime.now(timezone.utc).isoformat(),
                )
                if provenance_dir is not None:
                    _write_provenance(Path(provenance_dir), res)
                print(f"[Dynamics] Selected provider '{res.provider}' ({res.sfc_res_deg}° sfc + {res.ml_res_deg}° 137-ML)", flush=True)
                return res
            except Exception as exc:
                reason = str(exc)
                fallback_reasons[candidate] = reason
                attempts.append({"provider": candidate, "status": "error", "reason": reason})
                if strict_mode:
                    raise
                print(f"[Dynamics-Fallback] Provider '{candidate}' failed ({reason}); checking next fallback...", flush=True)
                continue

        if candidate == "ifs_0p1_sfc_cams_ml":
            try:
                ok, reason = probe_opendata_0p1(
                    cycle_date,
                    time_str=time_str,
                    max_step_h=max_step_h,
                    s3_base_url=s3_base,
                )
                if not ok:
                    fallback_reasons[candidate] = reason
                    attempts.append({"provider": candidate, "status": "unavailable", "reason": reason})
                    if strict_mode:
                        raise RuntimeError(f"Pinned dynamics provider '{candidate}' unavailable: {reason}")
                    print(f"[Dynamics-Fallback] Provider '{candidate}' unavailable ({reason}); checking next fallback...", flush=True)
                    continue

                download_opendata_0p1_sfc_daily(
                    cycle_date,
                    target_dir,
                    time_str=time_str,
                    leadtimes_hours=steps,
                    area=area,
                    max_workers=int(od_cfg.get("max_workers", 6)),
                    force=force,
                    s3_base_url=s3_base,
                )
                attempts.append({"provider": candidate, "status": "selected", "reason": "ok"})
                res = DynamicsResolution(
                    provider="ifs_0p1_sfc_cams_ml",
                    met_sfc_dir=target_dir,
                    met_sfc_prefix="ECMWF-IFS-0p1_sfc",
                    met_ml_dir=cams_path,
                    met_ml_prefix="ECMWF-CAMS_ml",
                    sfc_res_deg=0.1,
                    ml_res_deg=0.4,
                    fallback_reasons=fallback_reasons,
                    attempts=attempts,
                    resolved_at_utc=datetime.now(timezone.utc).isoformat(),
                )
                if provenance_dir is not None:
                    _write_provenance(Path(provenance_dir), res)
                print(f"[Dynamics] Selected provider '{res.provider}' (0.1° S3 sfc/soil + 0.4° CAMS 137-ML)", flush=True)
                return res
            except Exception as exc:
                reason = str(exc)
                fallback_reasons[candidate] = reason
                attempts.append({"provider": candidate, "status": "error", "reason": reason})
                if strict_mode:
                    raise
                print(f"[Dynamics-Fallback] Provider '{candidate}' failed ({reason}); checking next fallback...", flush=True)
                continue

        if candidate == "cams_0p4_ml":
            attempts.append({"provider": candidate, "status": "selected", "reason": "default CAMS fallback"})
            res = DynamicsResolution(
                provider="cams_0p4_ml",
                met_sfc_dir=cams_path,
                met_sfc_prefix="ECMWF-CAMS_sfc",
                met_ml_dir=cams_path,
                met_ml_prefix="ECMWF-CAMS_ml",
                sfc_res_deg=0.4,
                ml_res_deg=0.4,
                fallback_reasons=fallback_reasons,
                attempts=attempts,
                resolved_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            if provenance_dir is not None:
                _write_provenance(Path(provenance_dir), res)
            print("[Dynamics] Selected provider 'cams_0p4_ml' (0.4° CAMS 137-ML + sfc)", flush=True)
            return res

    raise RuntimeError(f"All dynamics providers failed: {fallback_reasons}")


def build_arg_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(
            prog="airblender fetch-dynamics",
            description="Resolve and download highest-available ECMWF dynamics with automatic fallback.",
        )
    parser.add_argument("--date", required=True, help="Cycle date (YYYY-MM-DD or YYYYMMDD)")
    parser.add_argument("--cams_dir", required=True, help="Directory holding CAMS NetCDFs")
    parser.add_argument("--out_dir", default=None, help="Directory for IFS outputs (default: <cams_dir>/../IFS)")
    parser.add_argument("--primary", default="ifs_mars_9km", choices=SUPPORTED_PROVIDERS)
    parser.add_argument("--mode", default="auto", choices=["auto", "pinned"])
    parser.add_argument("--leadtimes", nargs="+", type=int, default=[0, 3, 6, 9, 12, 15, 18, 21, 24])
    parser.add_argument("--area", nargs=4, type=float, default=[37.0, 41.0, 18.0, 61.0])
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    res = resolve_and_fetch_dynamics(
        args.date,
        cams_dir=args.cams_dir,
        out_dir=args.out_dir,
        dynamics_cfg={"mode": args.mode, "primary": args.primary},
        leadtimes_hours=args.leadtimes,
        area=args.area,
        provenance_dir=args.out_dir or Path(args.cams_dir).parent / "IFS",
        force=args.force,
    )
    print(json.dumps(res.to_dict(), indent=2))
