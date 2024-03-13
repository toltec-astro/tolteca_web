import re
from typing import Any


def make_toltec_raw_obs_uid(entry: dict[str:any]) -> str:
    """Return raw obs unique id."""
    return (
        f"{entry['master'].lower()}-{entry['obsnum']}"
        f"-{entry['subobsnum']}-{entry['scannum']}"
    )


def make_toltec_raw_obs_sweep_obs_uid(entry: dict[str:Any]) -> str:
    """Return raw obs unique id."""
    return (
        f"{entry['cal_master'].lower()}-{entry['cal_obsnum']}"
        f"-{entry['cal_subobsnum']}-{entry['cal_scannum']}"
    )


def parse_toltec_raw_obs_uid(uid: str) -> dict[str, Any]:
    """Return info parsed from raw obs unique id."""
    re_uid = r"(?P<master>ics|tcs)-(?P<obsnum>\d+)-(?P<subobsnum>\d+)-(?P<scannum>\d+)"
    dispatch_types = {"obsnum": int, "subobsnum": int, "scannum": int}
    m = re.match(re_uid, uid)
    if m is None:
        raise ValueError(f"invalid uid {uid}")
    d = m.groupdict()
    for k, v in d.items():
        d[k] = dispatch_types[k](v) if k in dispatch_types else v
    return d
