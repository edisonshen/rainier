"""Render 蔡森《多空转折一手抓》(scanned PDF) into per-unit page images.

The book is a scan (one image per page; the only text layer is a watermark ad),
so every reader model must work from page images. This splits the book into
reading units along its table of contents and writes:

    <out>/units.json              unit manifest (id, title, part, pdf pages)
    <out>/pages/<unit_id>/pNNN.png page images for that unit

The PDF and rendered pages are copyrighted — they stay local (gitignored).

Usage:
    uv run --with pymupdf python kb/caisen/render_pages.py \
        --pdf ~/Downloads/多空转折一手抓(高清).pdf --out kb/caisen/work
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pymupdf

# Chapter headings ("一、", "十二、", "附注", "附录", "后记", part headings)
# start a new unit; worked chart examples ("（一）…日线") stay with their chapter.
_UNIT_HEAD = re.compile(r"^([一二三四五六七八九十]+、|附注|附录|后记|\d+\s+\S+篇)")
_PART_HEAD = re.compile(r"^\d+\s+(\S+篇)")


def build_units(toc: list[list], page_count: int, max_pages: int = 12) -> list[dict]:
    heads: list[tuple[str, int, str]] = []
    part = ""
    for _lvl, title, page in toc:
        title = title.strip()
        m = _PART_HEAD.match(title)
        if m:
            part = m.group(1)
            continue
        if _UNIT_HEAD.match(title):
            if heads and heads[-1][1] == page:
                continue
            heads.append((title, page, part))
    if heads and heads[0][1] > 1:
        heads.insert(0, ("前言/序", 1, "前言"))
    units = []
    for i, (title, start, part_name) in enumerate(heads):
        end = max(start, heads[i + 1][1] - 1 if i + 1 < len(heads) else page_count)
        n_pages = end - start + 1
        n_chunks = -(-n_pages // max_pages)
        bounds = [start + n_pages * j // n_chunks for j in range(n_chunks + 1)]
        for j in range(n_chunks):
            suffix = chr(ord("a") + j) if n_chunks > 1 else ""
            units.append(
                {
                    "id": f"u{i:02d}{suffix}",
                    "title": title + (f" ({j + 1}/{n_chunks})" if suffix else ""),
                    "part": part_name,
                    "first_page": bounds[j],
                    "last_page": bounds[j + 1] - 1,
                }
            )
    return units


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--out", default=Path("kb/caisen/work"), type=Path)
    ap.add_argument("--dpi", default=110, type=int)
    ap.add_argument("--max-pages", default=12, type=int, help="split longer chapters into chunks")
    args = ap.parse_args()

    doc = pymupdf.open(args.pdf)
    units = build_units(doc.get_toc(), doc.page_count, args.max_pages)
    if not units:
        raise SystemExit("no units found in the PDF table of contents")

    pages_root = args.out / "pages"
    for unit in units:
        unit_dir = pages_root / unit["id"]
        unit_dir.mkdir(parents=True, exist_ok=True)
        for pno in range(unit["first_page"], unit["last_page"] + 1):
            target = unit_dir / f"p{pno:03d}.png"
            if not target.exists():
                doc[pno - 1].get_pixmap(dpi=args.dpi).save(target)
        unit["page_dir"] = str(unit_dir)

    (args.out / "units.json").write_text(
        json.dumps(units, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    total = sum(u["last_page"] - u["first_page"] + 1 for u in units)
    print(f"{len(units)} units, {total} pages -> {args.out}")


if __name__ == "__main__":
    main()
