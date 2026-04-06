from __future__ import annotations

import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


IMAGECONVERTPRO_APP_ID = "imageconvertpro"
WATERMARKPRO_APP_ID = "watermarkpro"
BRIDGE_MANIFEST_FLAG = "--bridge-manifest"


def _runtime_dirs(anchor_file: str) -> List[Path]:
    anchor_path = Path(anchor_file).resolve()
    anchor_dir = anchor_path.parent if anchor_path.is_file() else anchor_path

    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        workspace_dir = executable_dir.parent if executable_dir.name.lower() == "temp" else executable_dir
        dirs = [workspace_dir, workspace_dir / "Temp", executable_dir, anchor_dir]
    else:
        workspace_dir = anchor_dir.parent if anchor_dir.name.lower() == "temp" else anchor_dir
        dirs = [workspace_dir, workspace_dir / "Temp", anchor_dir]

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        dirs.append(Path(meipass).resolve())

    dirs.append(Path.cwd().resolve())

    unique_dirs: List[Path] = []
    seen = set()
    for directory in dirs:
        key = str(directory).lower()
        if key not in seen:
            seen.add(key)
            unique_dirs.append(directory)
    return unique_dirs


def workspace_root(anchor_file: str) -> Path:
    return _runtime_dirs(anchor_file)[0]


def bridge_root(anchor_file: str) -> Path:
    root = workspace_root(anchor_file) / "Temp" / "ProgramBridge"
    root.mkdir(parents=True, exist_ok=True)
    return root


def create_transfer_export_dir(anchor_file: str, sender_app: str, target_app: str) -> Path:
    directory = bridge_root(anchor_file) / "exports" / f"{sender_app}_to_{target_app}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def update_heartbeat(anchor_file: str, app_id: str) -> Path:
    heartbeat_dir = bridge_root(anchor_file) / "heartbeat"
    heartbeat_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = heartbeat_dir / f"{app_id}.json"
    payload = {
        "app_id": app_id,
        "pid": os.getpid(),
        "updated_at": time.time(),
    }
    heartbeat_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return heartbeat_path


def is_app_alive(anchor_file: str, app_id: str, max_age_seconds: float = 8.0) -> bool:
    heartbeat_path = bridge_root(anchor_file) / "heartbeat" / f"{app_id}.json"
    if not heartbeat_path.exists():
        return False

    try:
        payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    updated_at = float(payload.get("updated_at", 0.0))
    return (time.time() - updated_at) <= max_age_seconds


def _normalize_existing_files(paths: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()

    for raw_path in paths:
        if not raw_path:
            continue
        try:
            path = str(Path(raw_path).resolve())
        except Exception:
            continue
        if not os.path.isfile(path):
            continue
        key = os.path.normcase(os.path.normpath(path))
        if key in seen:
            continue
        seen.add(key)
        normalized.append(path)

    return normalized


def enqueue_transfer(
    anchor_file: str,
    sender_app: str,
    target_app: str,
    file_paths: Sequence[str],
    payload: Optional[dict] = None,
) -> Optional[Path]:
    files = _normalize_existing_files(file_paths)
    if not files:
        return None

    inbox_dir = bridge_root(anchor_file) / "inbox" / target_app
    inbox_dir.mkdir(parents=True, exist_ok=True)

    transfer_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    manifest_path = inbox_dir / f"{transfer_id}.json"
    temp_path = manifest_path.with_suffix(".json.tmp")

    manifest = {
        "transfer_id": transfer_id,
        "sender_app": sender_app,
        "target_app": target_app,
        "created_at": time.time(),
        "files": files,
        "payload": payload or {},
    }

    temp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temp_path, manifest_path)
    return manifest_path


def pop_pending_transfers(anchor_file: str, target_app: str) -> List[dict]:
    inbox_dir = bridge_root(anchor_file) / "inbox" / target_app
    inbox_dir.mkdir(parents=True, exist_ok=True)

    transfers: List[dict] = []
    for manifest_path in sorted(inbox_dir.glob("*.json")):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        files = _normalize_existing_files(payload.get("files", []))
        payload["files"] = files
        transfers.append(payload)

        try:
            manifest_path.unlink()
        except Exception:
            pass

    return transfers


def collect_launch_paths(args: Sequence[str]) -> List[str]:
    collected: List[str] = []
    index = 0

    while index < len(args):
        current = args[index]
        if current == BRIDGE_MANIFEST_FLAG:
            index += 1
            if index < len(args):
                manifest_path = Path(args[index]).resolve()
                if manifest_path.exists():
                    try:
                        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except Exception:
                        payload = {}
                    collected.extend(payload.get("files", []))
            index += 1
            continue

        collected.append(current)
        index += 1

    return collected


def find_companion_target(anchor_file: str, exe_names: Sequence[str], script_names: Sequence[str]) -> Optional[Path]:
    runtime_dirs = _runtime_dirs(anchor_file)

    for exe_name in exe_names:
        for directory in runtime_dirs:
            candidate = directory / exe_name
            if candidate.exists():
                return candidate

    for directory in _runtime_dirs(anchor_file):
        for script_name in script_names:
            candidate = directory / script_name
            if candidate.exists():
                return candidate
    return None


def get_python_launch_command() -> List[str]:
    if not getattr(sys, "frozen", False):
        return [sys.executable or "python"]

    py_launcher = shutil.which("py")
    if py_launcher:
        return [py_launcher, "-3"]

    python_cmd = shutil.which("python")
    if python_cmd:
        return [python_cmd]

    return ["python"]
