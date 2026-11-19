import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def _cpu_name() -> str:
    """Human-readable CPU name, e.g. 'Apple M2'; falls back to platform.processor()."""
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def device_info() -> dict:
    """
    Returns
    -------
    a dict describing the machine and software the timings were measured on:
    CPU name, number of logical CPUs, architecture, OS, Python and numpy versions.
    No host or user names are included, so the result can be published.
    """
    return {
        "cpu": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def save_costs(path: Path, costs: dict, repeats: int, params: dict | None = None) -> dict:
    """
    Parameters
    ----------
    path : JSON file the record is written to (overwritten if it exists)
    costs : measured costs, e.g. absolute times per sample and the ratios c, c'
    repeats : number of repeats each timing was taken over
    params : further settings the timings depend on (e.g. sample size n, M)

    Returns
    -------
    the saved record: {"timestamp", "repeats", "params", "costs", "device"},
    where "device" comes from device_info()
    """
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "repeats": repeats,
        "params": params or {},
        "costs": {key: float(value) for key, value in costs.items()},
        "device": device_info(),
    }
    Path(path).write_text(json.dumps(record, indent=2))
    return record


def load_costs(path: Path) -> dict | None:
    """
    Parameters
    ----------
    path : JSON file written by save_costs

    Returns
    -------
    the "costs" dict of the saved record, or None if the file does not exist
    """
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())["costs"]
