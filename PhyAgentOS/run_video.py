"""Cut the video a Skill runtime recorded during one task into a single mp4.

The ReKep perception node encodes its camera stream straight into mp4
segments named ``<start_wall_ns>.mp4`` under ``$REKEP_RUNTIME_ROOT``
(default ``~/.paos-rekep``) ``/run/rekep/recordings/<session>/``; a segment
ends at its last write (file mtime). ``paos agent -m`` calls
``build_run_video`` after the task so the task's clip exists without any test
harness. Nothing here talks to a Gateway or sends actions.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

NS = 1_000_000_000
_ENCODE = ["-an", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]


def recordings_dir() -> Path:
    root = os.environ.get("REKEP_RUNTIME_ROOT")
    base = Path(root).expanduser() if root else Path.home() / ".paos-rekep"
    return base / "run" / "rekep" / "recordings"


def select_segments(root: Path, start_ns: int, end_ns: int) -> list[tuple[int, int, Path]]:
    """``(segment_start_ns, segment_end_ns, path)`` of segments overlapping the window."""
    segments = []
    for item in root.glob("*/*.mp4"):
        try:
            begin, end = int(item.stem), item.stat().st_mtime_ns
        except (ValueError, OSError):
            continue
        if begin <= end_ns and end >= start_ns:
            segments.append((begin, end, item))
    return sorted(segments)


def wait_until_idle(recordings: Path | None = None, *, quiet_s: float = 2.5, timeout_s: float = 15.0) -> None:
    """Wait until no segment has been written for quiet_s (the node records 10 s past execute_task)."""
    recordings = recordings or recordings_dir()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        newest = 0
        for item in recordings.glob("*/*.mp4"):
            try:
                newest = max(newest, item.stat().st_mtime_ns)
            except OSError:
                continue
        if time.time_ns() - newest > quiet_s * NS:
            return
        time.sleep(0.5)


def _run(command: list[str]) -> subprocess.CompletedProcess:
    # -nostdin and no inherited stdin: a background job reading the tty is stopped by SIGTTIN.
    return subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=600)


def build_run_video(
    output: Path, start_ns: int, end_ns: int, recordings: Path | None = None
) -> tuple[Path | None, str]:
    """Write ``output`` (an .mp4) with the recorded video between start_ns and end_ns.

    Returns ``(path, message)``. ``path`` is None when no video was written;
    the message is empty when the task simply had no recording and describes
    the problem otherwise.
    """
    recordings = recordings or recordings_dir()
    segments = select_segments(recordings, start_ns, end_ns) if recordings.is_dir() else []
    if not segments:
        return None, ""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None, "找不到 ffmpeg，未生成视频"
    output.parent.mkdir(parents=True, exist_ok=True)
    base = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    with tempfile.TemporaryDirectory(prefix="paos-video-") as directory:
        parts = []
        for index, (begin, end, path) in enumerate(segments):
            offset_s = max(0, start_ns - begin) / NS
            duration_s = (min(end_ns, end) - max(start_ns, begin)) / NS
            if duration_s <= 0:
                continue
            part = Path(directory) / f"part-{index:03d}.mp4"
            try:
                completed = _run([*base, "-ss", f"{offset_s:.3f}", "-i", str(path), "-t", f"{duration_s:.3f}", *_ENCODE, str(part)])
            except (OSError, subprocess.TimeoutExpired) as exc:
                return None, f"ffmpeg 运行失败: {type(exc).__name__}"
            if completed.returncode or not part.is_file():
                return None, f"ffmpeg 失败: {completed.stderr[-200:].strip()}"
            parts.append(part)
        if not parts:
            return None, ""
        if len(parts) == 1:
            shutil.move(str(parts[0]), output)
        else:
            listing = Path(directory) / "parts.txt"
            listing.write_text("".join(f"file '{part}'\n" for part in parts))
            try:
                completed = _run([*base, "-f", "concat", "-safe", "0", "-i", str(listing), *_ENCODE, str(output)])
            except (OSError, subprocess.TimeoutExpired) as exc:
                return None, f"ffmpeg 运行失败: {type(exc).__name__}"
            if completed.returncode:
                return None, f"ffmpeg 失败: {completed.stderr[-200:].strip()}"
    seconds = (min(end_ns, segments[-1][1]) - max(start_ns, segments[0][0])) / NS
    return output, f"约 {seconds:.0f} 秒"
