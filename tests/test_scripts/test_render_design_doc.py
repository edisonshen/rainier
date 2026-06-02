"""Unit tests for scripts/render-design-doc.py — the md→hub-HTML renderer.

Pinned by docs/DESIGN-docs-publisher.md §4 (Renderer). All tests are
deterministic and touch no network. The renderer is imported as a module so
we exercise its functions directly, plus a subprocess smoke test of the CLI.
"""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RENDERER = ROOT / "scripts" / "render-design-doc.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("render_design_doc", RENDERER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rd = _load_module()


SAMPLE = """\
# Sample Design Doc

A subtitle paragraph with **bold** and `inline code`.

## First Section

Some body text linking to [another doc](DESIGN-other.md#anchor) and to an
[external site](https://example.com/path).

| Col A | Col B |
|-------|-------|
| 1     | 2     |

```
fenced code with <angle> brackets
```

> a block quote line

## Second Section

- bullet one
- bullet two
"""


def test_style_markers_present():
    """The output embeds template.html's exact style — assert the load-bearing
    markers (teal accent + heading anchors)."""
    out = rd.render_doc(SAMPLE)
    assert "0f766e" in out, "missing teal accent color from template.html"
    assert "heading-anchor" in out, "missing ¶ heading-anchor styling"
    assert "nav.toc" in out, "missing sticky TOC styling"


def test_deterministic():
    """Same input -> byte-identical output across runs."""
    a = rd.render_doc(SAMPLE)
    b = rd.render_doc(SAMPLE)
    assert hashlib.sha256(a.encode()).hexdigest() == hashlib.sha256(
        b.encode()
    ).hexdigest()


def test_toc_from_h2_headings():
    """`##` headings become sections + TOC entries; `#` is the page title."""
    out = rd.render_doc(SAMPLE)
    assert "<title>Sample Design Doc</title>" in out
    assert '<h1>Sample Design Doc</h1>' in out
    assert 'section id="first-section"' in out
    assert 'section id="second-section"' in out
    # TOC links to both sections.
    assert 'href="#first-section"' in out
    assert 'href="#second-section"' in out


def test_table_and_code_render():
    out = rd.render_doc(SAMPLE)
    assert "<table>" in out and "<th>Col A</th>" in out and "<td>1</td>" in out
    assert "<pre><code>" in out
    # Code content is escaped.
    assert "&lt;angle&gt;" in out


def test_blockquote_and_lists():
    out = rd.render_doc(SAMPLE)
    assert "<blockquote>" in out
    assert "<ul>" in out and "<li>bullet one</li>" in out


def test_ordered_list_renders():
    """An ordered (`1.`) list must render to <ol> without crashing — regression
    for the IndexError that read group(2) of a single-group ordered regex,
    which would crash the whole publish run (all-or-nothing) on any numbered
    list (e.g. the allowlisted INVESTIGATION doc's 'Open questions')."""
    md = "# T\n\n## S\n\n1. first item\n2. second item\n3. third item\n"
    out = rd.render_doc(md)
    assert "<ol>" in out and "</ol>" in out
    assert "<li>first item</li>" in out
    assert "<li>second item</li>" in out
    assert "<li>third item</li>" in out


def test_md_fragment_link_rewritten():
    """`*.md#frag` -> `*.html#frag` preserving the fragment."""
    out = rd.render_doc(SAMPLE)
    assert 'href="DESIGN-other.html#anchor"' in out
    assert "DESIGN-other.md" not in out, "raw .md link should be rewritten"


def test_external_link_untouched():
    out = rd.render_doc(SAMPLE)
    assert 'href="https://example.com/path"' in out


def test_md_link_without_fragment_rewritten():
    out = rd.render_doc("# T\n\n[x](TASK-PLAN-foo.md) body.\n")
    assert 'href="TASK-PLAN-foo.html"' in out


def test_javascript_link_neutralized():
    """`javascript:` links render as plain text, never an <a href>."""
    md = "# T\n\nClick [here](javascript:alert(1)) now.\n"
    out = rd.render_doc(md)
    assert 'href="javascript' not in out
    # The label survives as plain text.
    assert "here" in out


def test_data_link_neutralized():
    md = "# T\n\nA [payload](data:text/html;base64,PHNjcmlwdD4=) link.\n"
    out = rd.render_doc(md)
    assert 'href="data:' not in out


def test_raw_html_escaped():
    """Raw HTML in the source must be escaped — no passthrough."""
    md = "# T\n\nA paragraph with <script>alert(1)</script> embedded.\n"
    out = rd.render_doc(md)
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;" in out


def test_uppercase_javascript_scheme_neutralized():
    """Scheme matching is case-insensitive + whitespace tolerant."""
    md = "# T\n\n[x](JavaScript:alert(1)) and [y]( javascript:alert(2)).\n"
    out = rd.render_doc(md)
    assert "href=\"JavaScript" not in out
    assert "href=\" javascript" not in out
    assert 'href="javascript' not in out


def test_embedded_control_char_scheme_neutralized():
    """`java<TAB>script:` / `java<LF>script:` must be neutralized — browsers
    strip C0 control chars from a URL before resolving its scheme, so an
    embedded-control-char href would otherwise execute as javascript:."""
    for ctrl in ("\t", "\n", "\x00"):
        href, safe = rd._rewrite_href(f"java{ctrl}script:alert(1)")
        assert safe is False, f"embedded {ctrl!r} bypass must be caught"
    # End-to-end through the renderer (TAB form; LF can't survive md line join).
    md = "# T\n\n[x](java\tscript:alert(1)) link.\n"
    out = rd.render_doc(md)
    assert "<a href=" not in out, "control-char javascript: must not become an <a>"


def test_cli_smoke(tmp_path: Path):
    """The CLI renders a file to stdout deterministically."""
    src = tmp_path / "DESIGN-x.md"
    src.write_text(SAMPLE)
    r1 = subprocess.run(
        [sys.executable, str(RENDERER), str(src)],
        capture_output=True,
        text=True,
    )
    assert r1.returncode == 0, r1.stderr
    assert "0f766e" in r1.stdout
    r2 = subprocess.run(
        [sys.executable, str(RENDERER), str(src)],
        capture_output=True,
        text=True,
    )
    assert r1.stdout == r2.stdout, "CLI output must be deterministic"


def test_cli_missing_file_nonzero(tmp_path: Path):
    r = subprocess.run(
        [sys.executable, str(RENDERER), str(tmp_path / "nope.md")],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
