"""Mapping-rule DSL parser for atmospheric species and aerosol mode/bin transforms."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

_TERM_RE = re.compile(r"(-?\s*\.?[0-9]+(?:\.[0-9]*)?(?:[Ee]\s*-?\s*[0-9]+)?)\*\[(\w+)\]")


@dataclass(frozen=True)
class IcBcMappingItem:
    """Associates a target->source linear mapping expression with a donor stream name."""

    mapping_rule_str: str
    output_stream: str


def parse_mapping_rule(rule: str) -> list[dict[str, object]]:
    """Parse `<wrf_var> -> c1*[src1] + c2*[src2]; scale` into multiplier records."""
    parts = [p.strip() for p in re.split(r"->|;", rule)]
    if len(parts) != 3 or not parts[0] or not parts[1]:
        raise ValueError(f"Malformed mapping rule (expected 'target -> c*[src]; scale'): {rule!r}")

    wrf_key = parts[0]
    exponent = float(parts[2])
    terms = _TERM_RE.findall(parts[1])
    if not terms:
        raise ValueError(f"No valid 'coeff*[source_var]' terms parsed in rule: {rule!r}")

    # Verify every additive token matched so typos like '2.01[SS_cs]' fail loudly.
    residual = _TERM_RE.sub("", parts[1]).replace("+", "").replace(" ", "")
    if residual:
        raise ValueError(f"Unparsed syntax fragment {residual!r} in mapping rule: {rule!r}")

    records: list[dict[str, object]] = []
    for raw_coeff, src_key in terms:
        coeff = float(raw_coeff.replace(" ", ""))
        records.append(
            {
                "merra_key": src_key,
                "donor_key": src_key,
                "wrf_key": wrf_key,
                "wrf_multiplier": coeff,
                "wrf_exponent": exponent,
            }
        )
    return records


def parse_mapping_rules(rules: Iterable[str]) -> list[dict[str, object]]:
    """Parse a sequence of mapping rule strings."""
    parsed: list[dict[str, object]] = []
    for rule in rules:
        parsed.extend(parse_mapping_rule(rule))
    return parsed


def get_unique_wrf_keys_from_mappings(mappings: Iterable[IcBcMappingItem]) -> list[str]:
    """Return ordered unique target WRF variable names referenced by `mappings`."""
    unique_keys: list[str] = []
    for mapping in mappings:
        for item in parse_mapping_rule(mapping.mapping_rule_str):
            wrf_key = str(item["wrf_key"])
            if wrf_key not in unique_keys:
                unique_keys.append(wrf_key)
    return unique_keys
