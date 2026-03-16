from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Optional


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(
    root: str | Path,
    include_suffixes: Optional[Iterable[str]] = None,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Hash a directory tree deterministically.

    The digest is computed over (relative_path, file_bytes) pairs in sorted order.
    This is intended for run provenance (so you can prove which code produced a run).

    Args:
        root: Directory root.
        include_suffixes: If provided, only files whose suffix is in the iterable are
            included (e.g., [".py", ".json"]). If None, all files are included.
        chunk_size: Read chunk size for large files.
    """

    root_p = Path(root)
    if not root_p.exists() or not root_p.is_dir():
        raise FileNotFoundError(f"Not a directory: {root_p}")

    suffixes = set(include_suffixes) if include_suffixes is not None else None

    files = [p for p in root_p.rglob("*") if p.is_file()]
    files.sort(key=lambda p: str(p.relative_to(root_p)).replace('\\\\', '/'))

    h = hashlib.sha256()
    for p in files:
        if suffixes is not None and p.suffix not in suffixes:
            continue
        rel = str(p.relative_to(root_p)).replace('\\\\', '/')
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        with p.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()
