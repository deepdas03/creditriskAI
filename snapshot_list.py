# snapshot_list.py
"""
Helper utilities to list and inspect timestamped report snapshots saved under output/reports/.
Used by the Streamlit app to provide a 'Review past reports' page.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


def _read_meta(folder: Path) -> Dict[str, Any]:
    meta_path = folder / "meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception:
            return {"error": "meta_read_error"}
    else:
        files = list(folder.glob("*"))
        return {"inferred_files": [p.name for p in files]}


def list_snapshots(base_dir: str = "output/reports") -> List[Dict[str, Any]]:
    """
    Returns a list of snapshots with basic metadata, newest first.
    Each item: { "timestamp": "YYYYMMDD_HHMMSS", "path": "<fullpath>", "meta": {...} }
    """
    base = Path(base_dir)
    if not base.exists():
        return []
    entries = []
    for p in base.iterdir():
        if p.is_dir():
            ts = p.name
            try:
                parsed = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                friendly = parsed.isoformat()
            except Exception:
                friendly = ts
            meta = _read_meta(p)
            entries.append({
                "timestamp": ts,
                "friendly_ts": friendly,
                "path": str(p.resolve()),
                "meta": meta
            })
    entries.sort(key=lambda x: x["timestamp"], reverse=True)
    return entries


def get_snapshot_files(ts_folder: str) -> Dict[str, Any]:
    """
    Given a folder path (or relative timestamp under output/reports), return dict of files and sizes.
    """
    p = Path(ts_folder)
    if not p.exists():
        alt = Path("output/reports") / ts_folder
        if alt.exists():
            p = alt
        else:
            return {"error": "folder_not_found", "path": ts_folder}
    files = []
    for f in sorted(p.iterdir(), key=lambda x: x.name):
        if f.is_file():
            try:
                files.append({"name": f.name, "size_bytes": f.stat().st_size})
            except Exception:
                files.append({"name": f.name, "size_bytes": None})
    return {"path": str(p.resolve()), "files": files, "meta": _read_meta(p)}
