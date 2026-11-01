"""Materialise the raw Meituan challenge CSVs under ``data/raw/`` (idempotent).

The challenge tables are shipped under ``data/`` (or the repository root); this
stage copies them into the canonical ``data/raw/`` location so later stages can
rely on :data:`meituan_eta.config.RAW_WAYBILL` / ``RAW_WAVES``. The originals are
never moved or rewritten, and the big waybill table may also be recovered from
its ``.zip`` archive when only the archive is present.
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

from meituan_eta.config import RAW_WAVES, RAW_WAYBILL, common_argparser

WAYBILL_NAME: str = Path(RAW_WAYBILL).name
REQUIRED_FILES: tuple[str, ...] = (
    WAYBILL_NAME,
    Path(RAW_WAVES).name,
    "dispatch_rider_meituan.csv",
    "dispatch_waybill_meituan.csv",
)
# Search order mandated by the plan: canonical raw dir, then data/, then repo root.
SEARCH_DIRS: tuple[Path, ...] = (Path("data/raw"), Path("data"), Path("."))
ZIP_CANDIDATES: tuple[Path, ...] = (
    Path("data") / f"{WAYBILL_NAME}.zip",
    Path(f"{WAYBILL_NAME}.zip"),
)

_MISSING_POLICY = (
    "Place the challenge files under data/ or the repository root — the "
    "CC BY-NC 4.0 licence forbids redistributing them."
)


def _find_source(name: str) -> Path | None:
    """Return the first existing file for ``name`` following :data:`SEARCH_DIRS`."""
    for directory in SEARCH_DIRS:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _extract_waybill_from_zip(destination: Path) -> str | None:
    """Extract the waybill CSV from the first available zip; return the member name."""
    archive = next((path for path in ZIP_CANDIDATES if path.is_file()), None)
    if archive is None:
        return None
    with zipfile.ZipFile(archive) as handle:
        member = next(
            (
                info
                for info in handle.infolist()
                if PurePosixPath(info.filename).name == WAYBILL_NAME
                and "__MACOSX" not in PurePosixPath(info.filename).parts
            ),
            None,
        )
        if member is None:
            return None
        with handle.open(member) as source, destination.open("wb") as sink:
            shutil.copyfileobj(source, sink, length=1024 * 1024)
    return member.filename


def _materialise(name: str, out_dir: Path) -> str:
    """Ensure ``out_dir/name`` exists; return the human-readable status line."""
    destination = out_dir / name
    if destination.is_file() and destination.stat().st_size > 0:
        return f"[OK] {name} (already present, skipped)"

    source = _find_source(name)
    if source is not None:
        shutil.copy2(source, destination)
        return f"[COPY] {name} <- {source}"

    if name == WAYBILL_NAME:
        member = _extract_waybill_from_zip(destination)
        if member is not None:
            return f"[UNZIP] {name} <- {member}"

    return ""  # empty string signals "not materialised"


def main() -> int:
    parser = common_argparser("Materialise the raw Meituan challenge CSVs into data/raw/.")
    parser.add_argument("--out-dir", default="data/raw", help="destination directory for the CSVs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    for name in REQUIRED_FILES:
        status = _materialise(name, out_dir)
        if not status:
            missing.append(name)
            continue
        print(status)

    if missing:
        print(f"ERROR: could not find required challenge file(s): {', '.join(missing)}", file=sys.stderr)
        print(f"Expected filenames: {', '.join(REQUIRED_FILES)}", file=sys.stderr)
        print("Searched data/raw/, data/, and the repository root (plus the waybill .zip).", file=sys.stderr)
        print(_MISSING_POLICY, file=sys.stderr)
        return 1

    print(f"[OK] all {len(REQUIRED_FILES)} raw tables present under {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
