# # import os
# # import platform
# # from datetime import datetime
# # from pathlib import Path

# # def get_creation_datetime(path: str) -> datetime:
# #     """
# #     Return the creation datetime of the given file.
# #     On Windows, this is the real creation time.
# #     On macOS, uses st_birthtime if available.
# #     On Linux/other UNIX, falls back to last metadata change time.
# #     """
# #     p = Path(path)
# #     if not p.exists():
# #         raise FileNotFoundError(f"No such file: {path!r}")

# #     stat = p.stat()
# #     system = platform.system()

# #     if system == 'Windows':
# #         print("Windows----if")
# #         # On Windows, st_ctime is creation time
# #         ts = stat.st_ctime
# #     elif system == 'Darwin':
# #         print("Windows----elif")
# #         # On macOS, st_birthtime is creation time (if supported)
# #         ts = getattr(stat, "st_birthtime", stat.st_mtime)
# #     else:
# #         print("Windows----else")
# #         # On most UNIX (Linux, etc), no birth time—fall back to metadata change
# #         ts = stat.st_ctime

# #     return datetime.fromtimestamp(ts)

# # # Example usage:
# # if __name__ == "__main__":
# #     # filepath = "C:\\1-All-files\\IKONIC\\project-files\\Nik-Aerospace-RAG\\main-repo\\RAG_input\\Aero Accessories - Jun 2022.pdf"
# #     filepath = "C:\\1-All-files\\IKONIC\\project-files\\Nik-Aerospace-RAG\\main-repo\\RAG_input\\Project Wolverine - Redacted CIM Pages_Unlocked - Extract_removed.pdf"
# #     created = get_creation_datetime(filepath)
# #     print(f"File {filepath!r} was created on {created:%Y-%m-%d %H:%M:%S}")


# import os
# import platform
# import subprocess
# from datetime import datetime
# from pathlib import Path

# def _birthtime_via_stat(path: Path) -> float|None:
#     """
#     On Linux: run `stat -c %w` to get the birth time (or '-' if unknown).
#     Returns POSIX timestamp or None.
#     """
#     try:
#         output = subprocess.check_output(
#             ["stat", "-c", "%w", str(path)],
#             stderr=subprocess.DEVNULL,
#             text=True
#         ).strip()
#         if output != "-" and output:
#             # stat outputs like "2025-06-27 14:02:31.000000000 +0000"
#             # datetime.fromisoformat will parse the date portion.
#             # We split off the fractional & timezone for simplicity.
#             ts_str = output.split('.')[0]
#             dt = datetime.fromisoformat(ts_str)
#             return dt.timestamp()
#     except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
#         pass
#     return None

# def get_creation_datetime(path: str) -> datetime:
#     """
#     Return the file’s creation datetime if available, otherwise
#     falls back to a best guess (metadata change time or modification time).
#     """
#     p = Path(path)
#     if not p.exists():
#         raise FileNotFoundError(f"No such file: {path!r}")

#     stat = p.stat()
#     system = platform.system()
#     ts = None

#     if system == "Windows":
#         # On Windows, st_ctime is the creation time
#         ts = stat.st_ctime

#     else:
#         # Try macOS-style birthtime attribute
#         ts = getattr(stat, "st_birthtime", None)

#         if ts is None:
#             # Try Linux `stat -c %w`
#             ts = _birthtime_via_stat(p)

#         if ts is None:
#             # Finally, fallback to metadata change time
#             ts = stat.st_ctime

#     # If all else fails, ts will be non-null
#     return datetime.fromtimestamp(ts)

# # Example usage
# if __name__ == "__main__":
#     for f in ("C:\\1-All-files\\IKONIC\\project-files\\Nik-Aerospace-RAG\\main-repo\\RAG_input\\Project Wolverine - Redacted CIM Pages_Unlocked - Extract_removed.pdf"):
#         try:
#             created = get_creation_datetime(f)
#             print(f"{f!r} was created on {created:%Y-%m-%d %H:%M:%S}")
#         except FileNotFoundError:
#             print(f"{f!r} not found.")













#!/usr/bin/env python3
"""
pdf_true_creation_with_logs.py  (updated)

Now handles IndirectObject metadata entries.
"""

import sys
import logging
import platform
import subprocess
from pathlib import Path
from datetime import datetime

from PyPDF2 import PdfReader
from PyPDF2.generic import IndirectObject  # <— import this

# Configure logging (as before) …
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def _parse_pdf_creation_date(raw: str) -> datetime | None:
    logging.debug(f"Parsing PDF CreationDate string: {raw!r}")
    if not raw.startswith("D:"):
        logging.warning("PDF CreationDate string does not start with 'D:'")
        return None
    core = raw[2:16]  # YYYYMMDDHHmmSS
    try:
        dt = datetime.strptime(core, "%Y%m%d%H%M%S")
        logging.debug(f"Parsed datetime: {dt!r}")
        return dt
    except ValueError as e:
        logging.error(f"Failed to parse date portion {core!r}: {e}")
        return None

def get_pdf_metadata_creation(path: Path) -> datetime | None:
    logging.info(f"Attempting to read PDF metadata creation date from {path}")
    reader = PdfReader(str(path))
    info = getattr(reader, 'metadata', None) or getattr(reader, 'documentInfo', None)
    raw = None
    if info:
        raw = info.get("/CreationDate") or info.get("CreationDate")
    logging.debug(f"Raw metadata CreationDate before resolution: {raw!r}")

    # ==== NEW: resolve IndirectObject ====
    if isinstance(raw, IndirectObject):
        logging.debug("Found IndirectObject for CreationDate, resolving it now")
        raw = raw.get_object()
        logging.debug(f"Raw metadata CreationDate after resolution: {raw!r}")
    # =======================================

    if raw:
        return _parse_pdf_creation_date(raw)
    logging.info("No CreationDate field found in PDF metadata")
    return None

def _birthtime_via_stat(path: Path) -> float | None:
    """
    On Linux: run `stat -c %w` to get the birth time (or '-' if unknown).
    Returns a POSIX timestamp or None.
    """
    logging.info(f"Calling stat -c %w on {path}")
    try:
        out = subprocess.check_output(
            ["stat", "-c", "%w", str(path)],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        logging.debug(f"`stat -c %w` output: {out!r}")
        if out and out != "-":
            ts_str = out.split('.')[0]  # drop fractional seconds & timezone
            dt = datetime.fromisoformat(ts_str)
            logging.debug(f"Parsed birthtime from stat: {dt!r}")
            return dt.timestamp()
    except Exception as e:
        logging.warning(f"`stat -c %w` failed: {e}")
    return None

def get_filesys_creation(path: Path) -> datetime:
    """
    Fallback to filesystem creation/metadata-change time.
    """
    logging.info(f"Falling back to filesystem creation time for {path}")
    st = path.stat()
    logging.debug(f"stat result: st_ctime={st.st_ctime}, st_mtime={st.st_mtime}, st_birthtime={getattr(st, 'st_birthtime', None)}")
    sysname = platform.system()
    if sysname == "Windows":
        ts = st.st_ctime
        logging.debug("Using Windows st_ctime as creation time")
    elif sysname == "Darwin":
        ts = getattr(st, "st_birthtime", None) or st.st_mtime
        logging.debug("macOS: using st_birthtime if available, else st_mtime")
    else:
        birth_ts = _birthtime_via_stat(path)
        if birth_ts:
            ts = birth_ts
            logging.debug("Using Linux stat-born timestamp")
        else:
            ts = st.st_ctime
            logging.debug("Linux: falling back to st_ctime (inode change time)")
    return datetime.fromtimestamp(ts)

def main():
    if len(sys.argv) != 2:
        logging.error(f"Usage: {sys.argv[0]} /path/to/file.pdf")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        logging.error("Error: please point me at an existing .pdf file.")
        sys.exit(1)

    # 1) Try PDF metadata
    metadata_dt = get_pdf_metadata_creation(pdf_path)
    if metadata_dt:
        logging.info("Using PDF metadata CreationDate")
        print(metadata_dt.strftime("%Y-%m-%d %H:%M:%S"))
        return

    # 2) Fallback to filesystem
    fs_dt = get_filesys_creation(pdf_path)
    logging.info("Using filesystem timestamp as fallback")
    print(fs_dt.strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()
