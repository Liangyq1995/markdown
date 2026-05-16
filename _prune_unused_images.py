# -*- coding: utf-8 -*-
"""
Remove image files under picture/ and pictures/ not referenced by .md/.html docs.

Usage: python _prune_unused_images.py          # dry-run, list only
       python _prune_unused_images.py --apply # delete unused files

Treats as referenced:
- Local relative/absolute paths (substring match for picture/... or pictures/...)
- GitHub URLs for THIS repo (from .git/config origin): raw.githubusercontent.com,
  github.com/.../blob/..., github.com/.../raw/..., cdn.jsdelivr.net/gh/.../
- URL-decoded variants of the corpus for encoded paths.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent
GIT_CONFIG = ROOT / ".git" / "config"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".ico"}
SKIP_DIRS = {".git"}
ASSET_TOP = {"picture", "pictures"}


def parse_origin_owner_repo() -> tuple[str, str] | None:
    """Parse owner/repo from [remote \"origin\"] only (avoid matching other remotes)."""
    if not GIT_CONFIG.is_file():
        return None
    text = GIT_CONFIG.read_text(encoding="utf-8", errors="ignore")
    block = re.search(r'\[remote\s+"origin"\]([^\[]*)', text, re.I | re.DOTALL)
    if not block:
        return None
    b = block.group(1)
    m = re.search(r"url\s*=\s*git@github\.com:([^/]+)/([^\s]+)", b, re.I)
    if not m:
        m = re.search(r"url\s*=\s*https://github\.com/([^/]+)/([^\s]+)", b, re.I)
    if not m:
        return None
    repo = m.group(2).strip().removesuffix(".git")
    return m.group(1).lower(), repo.lower()


def is_doc(p: Path) -> bool:
    return p.suffix.lower() in {".md", ".html", ".htm", ".markdown"}


def collect_doc_text() -> str:
    parts: list[str] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            p = Path(dirpath) / name
            if not is_doc(p):
                continue
            try:
                raw = p.read_bytes()
            except OSError:
                continue
            for enc in ("utf-8", "utf-8-sig", "gbk", "cp936"):
                try:
                    parts.append(raw.decode(enc))
                    break
                except UnicodeDecodeError:
                    continue
    return "\n".join(parts)


def normalize_asset_path(p: str) -> str:
    p = unquote(p.replace("\\", "/").strip())
    low = p.lower()
    if low.startswith("picture/"):
        return "picture/" + p[8:]
    if low.startswith("pictures/"):
        return "pictures/" + p[9:]
    return p


def extract_github_asset_paths(corpus: str, owner: str, repo: str) -> set[str]:
    o, r = re.escape(owner), re.escape(repo)
    patterns = [
        rf"raw\.githubusercontent\.com/{o}/{r}/[^/]+/((?:picture|pictures)/[^\s\"\'<>?#]+)",
        rf"github\.com/{o}/{r}/blob/[^/]+/((?:picture|pictures)/[^\s\"\'<>?#]+)",
        rf"github\.com/{o}/{r}/raw/[^/]+/((?:picture|pictures)/[^\s\"\'<>?#]+)",
        rf"cdn\.jsdelivr\.net/gh/{o}/{r}@[^/]+/((?:picture|pictures)/[^\s\"\'<>?#]+)",
    ]
    out: set[str] = set()
    for pat in patterns:
        for m in re.finditer(pat, corpus, flags=re.I):
            raw_p = m.group(1).rstrip(').,]"\'')
            out.add(normalize_asset_path(raw_p))
    return out


def list_images() -> list[Path]:
    out: list[Path] = []
    for top in ASSET_TOP:
        base = ROOT / top
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                out.append(p)
    return out


def rel_posix(p: Path) -> str:
    return normalize_asset_path(p.relative_to(ROOT).as_posix())


def is_referenced(rel: str, corpus: str, corpus_decoded: str, gh_paths: set[str]) -> bool:
    rel_n = normalize_asset_path(rel)
    if rel_n in gh_paths:
        return True
    rel_bs = rel_n.replace("/", "\\")
    if rel_n in corpus or rel_bs in corpus:
        return True
    if rel_n in corpus_decoded or rel_bs in corpus_decoded:
        return True
    return False


def main() -> int:
    apply_delete = "--apply" in sys.argv
    remote = parse_origin_owner_repo()
    corpus = collect_doc_text()
    corpus_decoded = unquote(corpus)
    gh_paths: set[str] = set()
    if remote:
        owner, repo = remote
        gh_paths = extract_github_asset_paths(corpus, owner, repo)
        gh_paths = {normalize_asset_path(p) for p in gh_paths}

    images = list_images()
    unused: list[Path] = []
    for p in sorted(images, key=lambda x: str(x).lower()):
        rel = rel_posix(p)
        if is_referenced(rel, corpus, corpus_decoded, gh_paths):
            continue
        unused.append(p)

    print(f"origin repo: {remote}")
    print(f"paths from GitHub URLs (this repo): {len(gh_paths)}")
    for x in sorted(gh_paths):
        print(f"  gh: {x}")
    print(f"Total images: {len(images)}")
    print(f"Unused (not in any .md/.html local path + not in origin GitHub asset URLs): {len(unused)}")
    for p in unused:
        print(f"  {'DEL' if apply_delete else 'would delete'} {rel_posix(p)}")

    if not apply_delete:
        print("\nDry run only. Re-run with --apply to delete files.")
        return 0

    for p in unused:
        try:
            p.unlink()
        except OSError as e:
            print(f"  FAIL {p}: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
