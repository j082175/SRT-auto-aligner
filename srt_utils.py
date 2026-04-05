import re
from dataclasses import dataclass
from typing import List


@dataclass
class SRTSegment:
    index: int
    start: float   # seconds
    end: float     # seconds
    text: str


def _timestamp_to_seconds(ts: str) -> float:
    """'00:01:23,456' -> 83.456"""
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _seconds_to_timestamp(sec: float) -> str:
    """83.456 -> '00:01:23,456'"""
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = round((sec - int(sec)) * 1000)
    if ms >= 1000:
        ms -= 1000
        s += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt(path: str) -> List[SRTSegment]:
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    blocks = re.split(r"\n{2,}", content.strip())
    segments = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        time_match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            lines[1],
        )
        if not time_match:
            continue
        start = _timestamp_to_seconds(time_match.group(1))
        end = _timestamp_to_seconds(time_match.group(2))
        text = "\n".join(lines[2:]).strip()
        segments.append(SRTSegment(index=idx, start=start, end=end, text=text))
    return segments


def write_txt(segments: List[SRTSegment], path: str) -> None:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(
            f"{_seconds_to_timestamp(seg.start)} --> {_seconds_to_timestamp(seg.end)}"
        )
        lines.append(seg.text)
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_srt(segments: List[SRTSegment], path: str) -> None:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(
            f"{_seconds_to_timestamp(seg.start)} --> {_seconds_to_timestamp(seg.end)}"
        )
        lines.append(seg.text)
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
