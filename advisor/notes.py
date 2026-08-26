"""Free-text notes an RM keeps against each client.

Notes are the one thing in this app the user authors rather than derives, so
losing them is worse than losing a setting. They are too long to carry in the
query string, so they are written to a small JSON file beside the workbook —
that survives reruns, a client switch, and a dropped session.

It is not a database. On Streamlit Community Cloud the filesystem is ephemeral,
so a container restart clears the file; the UI says so. Every write failure is
swallowed and reported, because a read-only filesystem must degrade to
"notes work until you reload" rather than taking down the app.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

_FILENAME = "client_notes.json"


def _path() -> Path:
    override = os.environ.get("KADVISOR_NOTES_PATH")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent.parent / "data" / _FILENAME


def load() -> Dict[str, str]:
    """Every stored note, keyed by client id. Never raises."""
    try:
        raw = _path().read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def get(client_id: str) -> str:
    return load().get(client_id, "")


def save(client_id: str, text: str) -> Tuple[bool, Optional[str]]:
    """Store one client's note. Returns ``(ok, error_message)``."""
    notes = load()
    text = (text or "").strip()
    if text:
        notes[client_id] = text
    else:
        notes.pop(client_id, None)

    target = _path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file and replace, so an interrupted write cannot
        # leave a half-written file that would lose every other client's note.
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(notes, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(target)
        return True, None
    except OSError as exc:
        return False, str(exc)
