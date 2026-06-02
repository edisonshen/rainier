#!/usr/bin/env python3
"""render-design-doc.py — render a rainier `docs/*.md` to a self-contained,
hub-styled HTML page.

The output reuses the design-doc skill's `template.html` `<style>` VERBATIM
(teal #0f766e / warm paper, sticky `nav.toc`, `¶` heading anchors, sections
per `##`) so every published doc is consistently styled — the load-bearing
fix for the recurring "style not right" problem (docs/DESIGN-docs-publisher.md
§2/§3.1).

Pure stdlib. Deterministic. No network. Standalone:

    python3 scripts/render-design-doc.py docs/DESIGN-foo.md > out.html

Markdown subset supported (sufficient for the rainier doc corpus):
  - ATX headings `#`..`####`  (`##` => a `<section id=slug>` + TOC entry)
  - pipe tables with a `---|---` separator row
  - fenced code blocks ```...```
  - blockquotes `>`
  - unordered (`-`/`*`) and ordered (`1.`) lists (one level)
  - paragraphs
  - inline: **bold**, `code`, [text](link)

Safety (docs/DESIGN-docs-publisher.md §3.1, Codex P2):
  - ALL source text is `html.escape`d — no raw-HTML passthrough.
  - Link schemes `javascript:` / `data:` are neutralized (rendered as plain
    text, not an `<a href>`).

Link rewriting (Codex P1): intra-doc `*.md` links become `*.html`, PRESERVING
any `#fragment` (e.g. `DESIGN-foo.md#sec` -> `DESIGN-foo.html#sec`). External
/ absolute / anchor-only links are left alone.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Embedded style — copied VERBATIM from the design-doc skill's template.html
# `<style>` block. Kept inline (not read from ~/.claude at runtime) so the
# renderer is standalone + deterministic + testable without that file present.
# If template.html's style changes, re-sync this block.
# ---------------------------------------------------------------------------
TEMPLATE_STYLE = """\
  :root {
    color-scheme: light dark;
    --bg: #fbfaf7;
    --paper: #fffefb;
    --surface: #f4f1ea;
    --surface-2: #ede8dc;
    --fg: #1d2024;
    --muted: #626b73;
    --subtle: #8a9299;
    --border: #ded7ca;
    --border-strong: #c9bfad;
    --accent: #0f766e;
    --accent-rgb: 15, 118, 110;
    --code-bg: #efebe2;
    --row-alt: #f6f3ec;
    --shadow: 0 1px 2px rgba(36, 32, 24, 0.04), 0 12px 28px rgba(36, 32, 24, 0.06);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #101312;
      --paper: #171b19;
      --surface: #1d2320;
      --surface-2: #242b27;
      --fg: #e9e5dc;
      --muted: #a2aaa4;
      --subtle: #727c75;
      --border: #303832;
      --border-strong: #465047;
      --accent: #7bd9c7;
      --accent-rgb: 123, 217, 199;
      --code-bg: #222821;
      --row-alt: #1b211e;
      --shadow: 0 1px 2px rgba(0, 0, 0, 0.22), 0 18px 40px rgba(0, 0, 0, 0.24);
    }
  }
  html {
    background: var(--bg);
    color: var(--fg);
    scroll-behavior: smooth;
    text-rendering: optimizeLegibility;
    -webkit-font-smoothing: antialiased;
    font-kerning: normal;
  }
  body { background: var(--bg); color: var(--fg); }
  body {
    font: 16px/1.68 ui-sans-serif, -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
    display: grid;
    grid-template-columns: 250px minmax(0, 780px);
    column-gap: 64px;
    max-width: 1180px;
    margin: 0 auto;
    padding: 56px 32px 112px;
  }
  body > header,
  body > section,
  body > footer { grid-column: 2; }
  header {
    border-bottom: 1px solid var(--border);
    padding: 0 0 28px;
    margin-bottom: 40px;
  }
  h1, h2, h3, h4 {
    font-kerning: normal;
    letter-spacing: 0;
    text-wrap: balance;
    scroll-margin-top: 32px;
  }
  h1 {
    font-size: clamp(2.35rem, 4vw, 3.7rem);
    line-height: 1.04;
    font-weight: 720;
    margin: 0 0 12px;
    max-width: 13ch;
  }
  h2 {
    font-size: 1.42rem;
    line-height: 1.22;
    margin: 72px 0 18px;
    font-weight: 680;
    padding-top: 2px;
  }
  section:first-of-type h2 { margin-top: 0; }
  h3 {
    font-size: 1.08rem;
    line-height: 1.34;
    margin: 36px 0 10px;
    font-weight: 650;
  }
  h4 {
    font-size: 0.98rem;
    line-height: 1.4;
    margin: 22px 0 8px;
    font-weight: 650;
    color: var(--accent);
  }
  .heading-anchor {
    opacity: 0;
    border-bottom: 0;
    color: var(--subtle);
    font-weight: 500;
    margin-left: 0.45rem;
    transition: opacity 140ms ease, color 140ms ease;
  }
  :is(h2, h3, h4):hover .heading-anchor,
  .heading-anchor:focus {
    opacity: 1;
    color: var(--accent);
  }
  .meta { color: var(--muted); font-size: 0.86rem; line-height: 1.55; }
  p, li { margin: 0 0 12px; }
  p { max-width: 72ch; }
  ul, ol { padding-left: 24px; }
  li > h4 { margin-top: 28px; }
  strong { font-weight: 680; }
  code, pre {
    font-family: "SF Mono", ui-monospace, Menlo, Consolas, monospace;
    font-size: 0.86em;
  }
  code {
    background: var(--code-bg);
    border: 1px solid rgba(var(--accent-rgb), 0.08);
    padding: 1px 5px;
    border-radius: 4px;
  }
  pre {
    background: var(--code-bg);
    border: 1px solid var(--border);
    padding: 18px 20px;
    border-radius: 8px;
    overflow-x: auto;
    line-height: 1.45;
  }
  pre code { background: transparent; padding: 0; }
  pre.diagram-block {
    position: relative;
    text-align: center;
    margin: 28px auto 40px;
    padding: 24px 24px 26px;
    background:
      linear-gradient(180deg, rgba(var(--accent-rgb), 0.07), transparent 55%),
      var(--paper);
    box-shadow: var(--shadow);
  }
  pre.diagram-block code {
    display: inline-block;
    min-width: max-content;
    text-align: left;
  }
  blockquote {
    border-left: 3px solid var(--accent);
    margin: 20px 0 24px;
    padding: 12px 18px;
    color: var(--fg);
    background: rgba(var(--accent-rgb), 0.08);
    border-radius: 0 8px 8px 0;
  }
  blockquote p:last-child { margin-bottom: 0; }
  table {
    border-collapse: separate;
    border-spacing: 0;
    width: 100%;
    margin: 24px 0 36px;
    font-size: 0.91rem;
    line-height: 1.52;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    background: var(--paper);
  }
  th, td {
    text-align: left;
    vertical-align: top;
    padding: 12px 14px;
    border-bottom: 1px solid var(--border);
  }
  th {
    font-weight: 650;
    color: var(--fg);
    background: var(--surface);
  }
  tbody tr:last-child td { border-bottom: 0; }
  tbody tr:nth-child(even) td { background: var(--row-alt); }
  tbody tr:hover td { background: rgba(var(--accent-rgb), 0.07); }
  td.rank { font-variant-numeric: tabular-nums; white-space: nowrap; font-weight: 600; }
  a { color: var(--accent); text-decoration: none; border-bottom: 1px solid transparent; }
  a:hover { border-bottom-color: var(--accent); }
  nav.toc {
    grid-column: 1;
    grid-row: 2 / span 20;
    position: sticky;
    top: 28px;
    align-self: start;
    background: var(--paper);
    border: 1px solid var(--border);
    padding: 18px 18px 16px;
    border-radius: 8px;
    font-size: 0.86rem;
    line-height: 1.45;
    box-shadow: var(--shadow);
  }
  nav.toc strong {
    display: block;
    margin-bottom: 10px;
    font-size: 0.74rem;
    text-transform: uppercase;
    color: var(--muted);
  }
  nav.toc ol { padding-left: 19px; margin: 0; }
  nav.toc li { margin: 5px 0; padding-left: 2px; }
  nav.toc p {
    margin: 14px 0 0;
    padding-top: 14px;
    border-top: 1px solid var(--border);
    font-size: 0.8rem;
    color: var(--muted);
  }
  .pill {
    display: inline-block;
    font-size: 0.72rem;
    line-height: 1.4;
    padding: 2px 7px;
    border-radius: 999px;
    margin-left: 6px;
    font-weight: 650;
    vertical-align: 0.08em;
  }
  .pill.steal,
  .pill.warn {
    background: rgba(var(--accent-rgb), 0.12);
    color: var(--accent);
    border: 1px solid rgba(var(--accent-rgb), 0.2);
  }
  .qbox {
    position: relative;
    border: 1px solid var(--border);
    padding: 18px 20px 18px 22px;
    border-radius: 8px;
    margin: 20px 0;
    background:
      linear-gradient(90deg, rgba(var(--accent-rgb), 0.11), transparent 34%),
      var(--paper);
    box-shadow: var(--shadow);
  }
  .qbox::before {
    content: "";
    position: absolute;
    left: -1px;
    top: 12px;
    bottom: 12px;
    width: 3px;
    border-radius: 999px;
    background: var(--accent);
  }
  .qbox .qid {
    color: var(--accent);
    font-weight: 720;
    font-family: "SF Mono", Menlo, monospace;
    font-size: 0.85rem;
    margin-bottom: 8px;
  }
  .layer {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 22px;
    margin: 18px 0;
    background: var(--paper);
    box-shadow: var(--shadow);
  }
  .layer h3 { margin-top: 0; }
  .layer .borrows {
    font-size: 0.86rem;
    color: var(--muted);
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border);
  }
  footer {
    margin-top: 72px;
    padding: 22px 0 0;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 0.86rem;
  }
  .sources { columns: 2; column-gap: 28px; font-size: 0.88rem; }
  .sources li { break-inside: avoid; margin-bottom: 6px; }
  @media (max-width: 1023px) {
    body {
      display: block;
      max-width: 820px;
      padding: 40px 24px 88px;
    }
    header { margin-bottom: 24px; }
    h1 { max-width: 15ch; }
    h2 { margin-top: 56px; }
    nav.toc {
      position: static;
      margin: 0 0 44px;
    }
  }
  @media (max-width: 720px) {
    body { padding: 32px 18px 72px; }
    h1 { font-size: 2.2rem; }
    table {
      display: block;
      overflow-x: auto;
      white-space: normal;
    }
    pre.diagram-block {
      text-align: left;
      padding: 20px 18px;
    }
    .sources { columns: 1; }
  }
  @media print {
    @page { margin: 0.75in; }
    :root {
      --bg: #ffffff;
      --paper: #ffffff;
      --surface: #ffffff;
      --surface-2: #ffffff;
      --fg: #111111;
      --muted: #555555;
      --subtle: #777777;
      --border: #cccccc;
      --accent: #111111;
      --accent-rgb: 17, 17, 17;
      --code-bg: #f3f3f3;
      --row-alt: #f7f7f7;
      --shadow: none;
    }
    html { scroll-behavior: auto; }
    body {
      display: block;
      max-width: none;
      padding: 0;
      font-size: 11pt;
      line-height: 1.5;
    }
    header, nav.toc, section, footer { break-inside: auto; }
    nav.toc {
      position: static;
      box-shadow: none;
      margin: 0 0 24pt;
      page-break-after: avoid;
    }
    h1 { font-size: 28pt; max-width: none; }
    h2 { font-size: 16pt; margin-top: 28pt; page-break-after: avoid; }
    h3, h4 { page-break-after: avoid; }
    .heading-anchor { display: none; }
    a { color: inherit; border-bottom: 0; text-decoration: underline; }
    pre, table, blockquote, .qbox, .layer { break-inside: avoid; box-shadow: none; }
    table { font-size: 9pt; }
    th, td { padding: 7pt 8pt; }
    footer { margin-top: 32pt; }
  }
  .anchor-skip { position: absolute; left: -9999px; }
"""

# Unsafe URL schemes — links using these are rendered as plain text, never an
# <a href> (Codex P2). Browsers strip ALL control chars (incl. embedded TAB/LF)
# from a URL before resolving its scheme, so `java\tscript:` / `java\nscript:`
# execute as `javascript:`. We mirror that: strip every C0 control char from the
# scheme prefix before matching (see _strip_url_ctrl), so leading- AND embedded-
# control-char bypasses (`JavaScript:` / ` javascript:` / `java\tscript:`) are
# all caught.
_UNSAFE_SCHEME_RE = re.compile(r"^(javascript|data|vbscript)\s*:", re.I)
# C0 controls + space — what browsers drop from a URL before scheme resolution.
_URL_CTRL_RE = re.compile(r"[\x00-\x20]")


def _strip_url_ctrl(href: str) -> str:
    """Remove C0 control chars + spaces, mirroring browser URL normalization.
    Used only for the unsafe-scheme check so `java\\tscript:` can't slip past."""
    return _URL_CTRL_RE.sub("", href)


def slugify(text: str) -> str:
    """Stable URL slug for a heading. Deterministic; collisions get a -N suffix
    handled by the caller."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s-]+", "-", text)
    return text.strip("-") or "section"


def _rewrite_href(href: str) -> tuple[str, bool]:
    """Return (href_or_text, is_safe).

    - Unsafe scheme -> (original, False): caller renders as plain text.
    - Intra-doc `*.md[#frag]` -> rewritten `*.html[#frag]`.
    - Everything else -> unchanged.
    """
    if _UNSAFE_SCHEME_RE.match(_strip_url_ctrl(href)):
        return href, False

    # Only rewrite relative .md links (not external URLs, not absolute paths
    # with a scheme). A link is "external" if it has a scheme like http(s):.
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", href):
        # Has a scheme (http:, https:, mailto:, ...) — leave alone (already
        # passed the unsafe-scheme gate above).
        return href, True

    # Split off a fragment so we preserve `#section`.
    frag = ""
    base = href
    hash_idx = href.find("#")
    if hash_idx != -1:
        base = href[:hash_idx]
        frag = href[hash_idx:]  # includes the leading '#'

    if base.endswith(".md"):
        base = base[: -len(".md")] + ".html"
    return base + frag, True


# Inline span parsing: emit already-escaped HTML. We tokenize code spans first
# (their contents are NOT further processed), then bold, then links.
_CODE_SPAN_RE = re.compile(r"`([^`]+)`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")


def render_inline(text: str) -> str:
    """Render inline markdown to safe HTML. All literal text is html.escape'd;
    only the generated tags are raw."""
    # Protect code spans with placeholders so their contents aren't touched by
    # bold/link parsing.
    code_spans: list[str] = []

    def _stash_code(m: re.Match[str]) -> str:
        code_spans.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x00CODE{len(code_spans) - 1}\x00"

    text = _CODE_SPAN_RE.sub(_stash_code, text)

    # Links: stash as placeholders too (so their visible text gets bold/escape
    # handling but the href is controlled). Each match appends exactly one
    # entry to `links` and returns its placeholder.
    links: list[str] = []

    def _render_label(label: str) -> str:
        inner = html.escape(label)
        return _BOLD_RE.sub(r"<strong>\1</strong>", inner)

    def _stash_link(m: re.Match[str]) -> str:
        label, href = m.group(1), m.group(2).strip()
        new_href, safe = _rewrite_href(href)
        inner = _render_label(label)
        if not safe:
            # Unsafe scheme: render label + escaped URL as plain text — no <a>.
            payload = f"{inner} ({html.escape(href)})"
        else:
            payload = f'<a href="{html.escape(new_href, quote=True)}">{inner}</a>'
        links.append(payload)
        return f"\x00LINK{len(links) - 1}\x00"

    text = _LINK_RE.sub(_stash_link, text)

    # Escape everything that's left (literal text), then bold.
    text = html.escape(text)
    text = _BOLD_RE.sub(r"<strong>\1</strong>", text)

    # Restore link placeholders (their inner label still needs escape+bold).
    for i, payload in enumerate(links):
        text = text.replace(f"\x00LINK{i}\x00", payload)
    # Restore code placeholders.
    for i, payload in enumerate(code_spans):
        text = text.replace(f"\x00CODE{i}\x00", payload)
    return text


# ---------------------------------------------------------------------------
# Block parsing
# ---------------------------------------------------------------------------


def _is_table_sep(line: str) -> bool:
    s = line.strip()
    if "|" not in s and "-" not in s:
        return False
    cells = [c.strip() for c in s.strip("|").split("|")]
    return len(cells) >= 1 and all(re.fullmatch(r":?-+:?", c) for c in cells if c != "")


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def parse_markdown(md: str) -> tuple[str, str, list[tuple[str, str]], str]:
    """Parse markdown into (title, subtitle, toc, body_html).

    title    — text of the first `# ` heading (page <h1>/<title>).
    subtitle — first non-blank paragraph after the title (page meta), if any.
    toc      — list of (slug, heading-text) for each `## ` heading.
    body     — the rendered <section>...</section> HTML.
    """
    lines = md.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    title = ""
    subtitle = ""
    toc: list[tuple[str, str]] = []
    seen_slugs: dict[str, int] = {}
    out: list[str] = []  # body chunks
    section_open = False

    i = 0
    n = len(lines)

    def close_section() -> None:
        nonlocal section_open
        if section_open:
            out.append("</section>")
            section_open = False

    def ensure_section() -> None:
        """Open an implicit intro <section> for body content that appears before
        the first `##` (e.g. a doc opening with a list/table). Without this the
        block lands as a direct `body >` child, which the grid CSS only assigns
        to column 2 for header/section/footer — so pre-`##` content would spill
        into the left TOC column (Codex P2)."""
        nonlocal section_open
        if not section_open:
            out.append("<section>")
            section_open = True

    def uniq_slug(text: str) -> str:
        base = slugify(text)
        if base in seen_slugs:
            seen_slugs[base] += 1
            return f"{base}-{seen_slugs[base]}"
        seen_slugs[base] = 0
        return base

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Fenced code block.
        if stripped.startswith("```"):
            i += 1
            code_lines: list[str] = []
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # consume closing fence
            code = html.escape("\n".join(code_lines))
            ensure_section()
            out.append(f"<pre><code>{code}</code></pre>")
            continue

        # Blank line.
        if stripped == "":
            i += 1
            continue

        # Headings.
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            htext = m.group(2).strip()
            if level == 1:
                if not title:
                    title = htext
                else:
                    # Secondary h1 — treat as h2-ish; rare in our corpus.
                    close_section()
                    slug = uniq_slug(htext)
                    toc.append((slug, htext))
                    out.append(f'<section id="{slug}">')
                    section_open = True
                    out.append(_heading_html(2, htext, slug))
                i += 1
                continue
            if level == 2:
                close_section()
                slug = uniq_slug(htext)
                toc.append((slug, htext))
                out.append(f'<section id="{slug}">')
                section_open = True
                out.append(_heading_html(2, htext, slug))
                i += 1
                continue
            # h3 / h4
            slug = uniq_slug(htext)
            ensure_section()
            out.append(_heading_html(level, htext, slug))
            i += 1
            continue

        # Table: a header row followed by a separator row.
        if "|" in line and i + 1 < n and _is_table_sep(lines[i + 1]):
            header = _split_row(line)
            i += 2  # skip header + separator
            rows: list[list[str]] = []
            while i < n and "|" in lines[i] and lines[i].strip() != "":
                rows.append(_split_row(lines[i]))
                i += 1
            ensure_section()
            out.append(_table_html(header, rows))
            continue

        # Blockquote.
        if stripped.startswith(">"):
            quote_lines: list[str] = []
            while i < n and lines[i].strip().startswith(">"):
                quote_lines.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            inner = render_inline(" ".join(ql.strip() for ql in quote_lines))
            ensure_section()
            out.append(f"<blockquote><p>{inner}</p></blockquote>")
            continue

        # Lists.
        if re.match(r"^\s*([-*])\s+", line):
            items, i = _collect_list(lines, i, ordered=False)
            ensure_section()
            out.append(_list_html(items, ordered=False))
            continue
        if re.match(r"^\s*\d+\.\s+", line):
            items, i = _collect_list(lines, i, ordered=True)
            ensure_section()
            out.append(_list_html(items, ordered=True))
            continue

        # Paragraph — gather consecutive non-blank, non-special lines.
        para_lines = [line]
        i += 1
        while i < n:
            nxt = lines[i]
            ns = nxt.strip()
            if ns == "" or ns.startswith(("#", ">", "```")):
                break
            if re.match(r"^\s*([-*])\s+", nxt) or re.match(r"^\s*\d+\.\s+", nxt):
                break
            if "|" in nxt and i + 1 < n and _is_table_sep(lines[i + 1]):
                break
            para_lines.append(nxt)
            i += 1
        para_text = " ".join(pl.strip() for pl in para_lines)
        rendered = render_inline(para_text)
        if not subtitle and not toc:
            # First paragraph before any `##` becomes the page subtitle/meta.
            subtitle = rendered
        else:
            ensure_section()
            out.append(f"<p>{rendered}</p>")

    close_section()
    return title, subtitle, toc, "\n".join(out)


def _heading_html(level: int, text: str, slug: str) -> str:
    inner = render_inline(text)
    label = html.escape(re.sub(r"\s+", " ", text).strip(), quote=True)
    anchor = (
        f'<a class="heading-anchor" href="#{slug}" '
        f'aria-label="Link to {label}">¶</a>'
    )
    return f'<h{level} id="{slug}">{inner} {anchor}</h{level}>'


def _table_html(header: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{render_inline(c)}</th>" for c in header)
    body_rows = []
    for r in rows:
        # Pad/truncate to header width.
        cells = (r + [""] * len(header))[: len(header)]
        tds = "".join(f"<td>{render_inline(c)}</td>" for c in cells)
        body_rows.append(f"<tr>{tds}</tr>")
    return (
        "<table>\n<thead>\n<tr>"
        + th
        + "</tr>\n</thead>\n<tbody>\n"
        + "\n".join(body_rows)
        + "\n</tbody>\n</table>"
    )


def _collect_list(
    lines: list[str], i: int, *, ordered: bool
) -> tuple[list[str], int]:
    items: list[str] = []
    # Both patterns capture the item TEXT in group 1 so the same access works.
    # (The ordered pattern previously had only one group but read group(2),
    # raising IndexError on every `1.` list — see test_ordered_list_renders.)
    pat = r"^\s*\d+\.\s+(.*)$" if ordered else r"^\s*[-*]\s+(.*)$"
    n = len(lines)
    while i < n:
        m = re.match(pat, lines[i])
        if not m:
            break
        items.append(m.group(1))
        i += 1
    return items, i


def _list_html(items: list[str], *, ordered: bool) -> str:
    tag = "ol" if ordered else "ul"
    lis = "".join(f"<li>{render_inline(it)}</li>" for it in items)
    return f"<{tag}>{lis}</{tag}>"


def render_doc(md: str) -> str:
    """Render markdown source to a full self-contained HTML page."""
    title, subtitle, toc, body = parse_markdown(md)
    page_title = title or "Document"

    toc_items = "\n".join(
        f'    <li><a href="#{slug}">{render_inline(text)}</a></li>' for slug, text in toc
    )
    toc_block = (
        '<nav class="toc" aria-label="Contents">\n'
        "  <strong>Contents</strong>\n"
        "  <ol>\n" + toc_items + "\n  </ol>\n</nav>"
        if toc
        else ""
    )

    meta_html = f'  <div class="meta">{subtitle}</div>\n' if subtitle else ""

    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f"<title>{html.escape(page_title)}</title>\n"
        "<style>\n" + TEMPLATE_STYLE + "</style>\n"
        "</head>\n"
        "<body>\n"
        "<header>\n"
        f"  <h1>{html.escape(page_title)}</h1>\n"
        f"{meta_html}"
        "</header>\n\n"
        f"{toc_block}\n\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write("usage: render-design-doc.py <doc.md>\n")
        return 2
    src = Path(argv[1])
    if not src.is_file():
        sys.stderr.write(f"error: not a file: {src}\n")
        return 1
    md = src.read_text(encoding="utf-8")
    sys.stdout.write(render_doc(md))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
