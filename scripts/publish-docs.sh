#!/bin/bash
# publish-docs.sh — render the curated rainier `docs/*.md` doc set to hub-styled
# HTML and sync it into the fengshen-site Astro `public/projdoc/rainier/` tree,
# commit + push so Cloudflare Pages auto-deploys.
#
# Serves the curated docs at https://fengshen.dev/projdoc/rainier/ (the same
# convention that already serves the manually-placed recursive-research-system.html).
# Modeled on scripts/publish-dashboard.sh; see docs/DESIGN-docs-publisher.md.
#
# ASCII flow (ALL-OR-NOTHING — a render failure NEVER dirties the target):
#
#   docs/*.md (curated allowlist, .md only)
#         │  render-design-doc.py  (each -> staging dir OUTSIDE target)
#         ▼
#   $STAGING/<basename>.html  +  index.html (git-date sorted)
#         │  every doc rendered OK ?
#    ┌────┴─────┐
#  no│          │yes
#    ▼          ▼   cp staged -> public/projdoc/rainier/  (upsert-only, no delete)
#  exit 1   git add public/projdoc/rainier/  →  diff? → commit + push
#  (target   (no diff → no-op; recover unpushed @{u}..HEAD)
#   pristine)
#
# The whole publish body runs under a SHARED publish lock (scripts/_publish_lock.py)
# keyed by the resolved target path, so it can't race the dashboard publishers
# against the same checkout (docs/DESIGN-docs-publisher.md §3.2).
#
# Environment overrides (defaults match the operator's machine):
#   DOCS_SOURCE_DIR           markdown source dir ($HOME/projects/rainier/docs)
#   DOCS_PUBLISH_TARGET_DIR   fengshen-site checkout ($HOME/projects/fengshen-site)
#   RAINIER_PYTHON            python3 to use (default: python3 on PATH)

set -euo pipefail

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '%s [publish-docs] %s\n' "$(ts)" "$*"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_DIR="${DOCS_SOURCE_DIR:-$HOME/projects/rainier/docs}"
TARGET_DIR="${DOCS_PUBLISH_TARGET_DIR:-$HOME/projects/fengshen-site}"
PY="${RAINIER_PYTHON:-python3}"
RENDERER="$SCRIPT_DIR/render-design-doc.py"
LOCK="$SCRIPT_DIR/_publish_lock.py"

# Published tree, relative to the target repo root.
DST_REL="public/projdoc/rainier"

# Re-exec the whole body under the shared lock unless we're already holding it.
# The lock is keyed by the RESOLVED target so dashboards + docs serialize.
if [ "${_PUBLISH_DOCS_LOCKED:-0}" != "1" ]; then
    if [ ! -d "$TARGET_DIR" ]; then
        log "ERROR target dir does not exist: $TARGET_DIR"
        exit 1
    fi
    RESOLVED_TARGET="$(cd "$TARGET_DIR" && pwd -P)"
    export _PUBLISH_DOCS_LOCKED=1
    exec "$PY" "$LOCK" "$RESOLVED_TARGET" -- bash "$0" "$@"
fi

# ---------------------------------------------------------------------------
# Locked body.
# ---------------------------------------------------------------------------

log "source=$SOURCE_DIR target=$TARGET_DIR"

if [ ! -d "$SOURCE_DIR" ]; then
    log "ERROR source docs dir missing: $SOURCE_DIR"
    exit 1
fi

if [ ! -d "$TARGET_DIR/.git" ]; then
    log "ERROR target is not a git checkout: $TARGET_DIR"
    exit 1
fi

# Bail if the target working tree has unrelated dirt — don't auto-stash.
# (Same guard as publish-dashboard.sh:105.)
if [ -n "$(git -C "$TARGET_DIR" status --porcelain)" ]; then
    log "ERROR target working tree dirty (uncommitted changes); refusing to publish"
    git -C "$TARGET_DIR" status --short >&2
    exit 1
fi

# Build the curated allowlist: .md files matching the doc prefixes. `.html`
# companions are deliberately ignored (no *.html.html). Sorted for determinism.
shopt -s nullglob
SRC_DOCS=()
for prefix in DESIGN TASK-PLAN TEST-SPEC SPIKE RESEARCH PLAN INVESTIGATION; do
    for f in "$SOURCE_DIR"/"$prefix"-*.md; do
        SRC_DOCS+=("$f")
    done
done
shopt -u nullglob

if [ "${#SRC_DOCS[@]}" -eq 0 ]; then
    log "no allowlisted docs found in $SOURCE_DIR — nothing to publish"
    exit 0
fi

# Sort the list deterministically (basenames are unique within docs/).
# `read` loop instead of array-from-subshell so this works on macOS bash 3.2
# (no `mapfile`) and avoids word-splitting surprises (shellcheck SC2207).
_sorted=()
while IFS= read -r _line; do
    _sorted+=("$_line")
done < <(printf '%s\n' "${SRC_DOCS[@]}" | sort)
SRC_DOCS=("${_sorted[@]}")

# Staging dir OUTSIDE the target tree (all-or-nothing: a render failure must
# never touch the target). Cleaned up on every exit path.
STAGING="$(mktemp -d "${TMPDIR:-/tmp}/publish-docs.XXXXXX")"
cleanup() { rm -rf "$STAGING"; }
trap cleanup EXIT

# Path-confinement helper: the published abspath must stay under the target's
# public/projdoc/rainier tree. Rejects traversal / symlink escape.
TARGET_ABS="$(cd "$TARGET_DIR" && pwd -P)"
CONFINE_ABS="$TARGET_ABS/$DST_REL"

# index.html entries accumulate here: "slug<TAB>title<TAB>date".
INDEX_TSV="$STAGING/.index.tsv"
: > "$INDEX_TSV"

cd "$TARGET_DIR"

# Render every doc into staging FIRST. Any failure aborts before we touch the
# target — the shared checkout stays pristine.
for src in "${SRC_DOCS[@]}"; do
    # Refuse symlinked source docs (don't follow them out of docs/).
    if [ -L "$src" ]; then
        log "ERROR refusing symlinked source doc: $src"
        exit 1
    fi
    base="$(basename "$src" .md)"

    # Path-confinement: reject any basename that isn't a plain filename.
    case "$base" in
        */*|*..*|"")
            log "ERROR unsafe doc basename: $base"
            exit 1
            ;;
    esac

    out="$STAGING/$base.html"
    # realpath of the FINAL destination must stay under the confine dir.
    final_abs="$CONFINE_ABS/$base.html"
    case "$final_abs" in
        "$CONFINE_ABS"/*) : ;;
        *)
            log "ERROR output path escapes target tree: $final_abs"
            exit 1
            ;;
    esac

    if ! "$PY" "$RENDERER" "$src" > "$out"; then
        log "ERROR render failed for $src — aborting (target left pristine)"
        exit 1
    fi

    # Title for the index = first `# ` heading, falling back to the basename.
    title="$(grep -m1 '^# ' "$src" | sed 's/^# //' || true)"
    [ -n "$title" ] || title="$base"
    # Last-commit date from git (NOT mtime), so identical content stays a true
    # no-op even after a `touch`. Fall back to a fixed string for untracked
    # source docs (e.g. fixtures) so the index remains deterministic.
    cdate="$(git -C "$SOURCE_DIR" log -1 --format=%cs -- "$src" 2>/dev/null || true)"
    [ -n "$cdate" ] || cdate="unpublished"
    printf '%s\t%s\t%s\n' "$base" "$title" "$cdate" >> "$INDEX_TSV"
done

# Build a deterministic index.html (sorted by basename) into staging.
"$PY" - "$INDEX_TSV" > "$STAGING/index.html" <<'PYEOF'
import html
import sys

rows = []
with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        slug, title, date = line.split("\t", 2)
        rows.append((slug, title, date))
rows.sort(key=lambda r: r[0])

items = []
for slug, title, date in rows:
    items.append(
        f'  <li><a href="{html.escape(slug, quote=True)}.html">'
        f"{html.escape(title)}</a>"
        f' <span class="meta">{html.escape(date)}</span></li>'
    )

print("<!doctype html>")
print('<html lang="en"><head><meta charset="utf-8">')
print('<meta name="viewport" content="width=device-width,initial-scale=1">')
print("<title>rainier docs</title>")
print("<style>")
print("body{font:16px/1.6 ui-sans-serif,-apple-system,sans-serif;"
      "max-width:760px;margin:0 auto;padding:56px 24px;"
      "background:#fbfaf7;color:#1d2024}")
print("h1{font-weight:720}")
print("a{color:#0f766e;text-decoration:none}a:hover{text-decoration:underline}")
print("ul{list-style:none;padding:0}li{margin:8px 0}")
print(".meta{color:#626b73;font-size:.82rem;margin-left:8px}")
print("</style></head><body>")
print("<h1>rainier docs</h1>")
print("<ul>")
for it in items:
    print(it)
print("</ul>")
print("</body></html>")
PYEOF

# All docs rendered. Now (and only now) sync staged HTML into the target tree.
# Upsert-only: copy our own outputs; NEVER delete pre-existing pages (protects
# the manually-placed recursive-research-system.html).
#
# Destination path-confinement (Codex P2): the earlier per-doc check is purely
# string-based, so a tracked symlink at public/projdoc or public/projdoc/rainier
# would let `cp` follow it and write OUTSIDE the repo (git would then see no
# published change). Reject any symlinked path component, then realpath-verify
# the created dir still resolves under the resolved target tree before copying.
for component in "public" "public/projdoc" "public/projdoc/rainier"; do
    if [ -L "$TARGET_ABS/$component" ]; then
        log "ERROR refusing symlinked published-tree component: $component"
        exit 1
    fi
done
mkdir -p "$CONFINE_ABS"
CONFINE_REAL="$(cd "$CONFINE_ABS" && pwd -P)"
case "$CONFINE_REAL/" in
    "$TARGET_ABS"/public/projdoc/rainier/) : ;;
    *)
        log "ERROR published dir escapes target tree: $CONFINE_REAL"
        exit 1
        ;;
esac
for f in "$STAGING"/*.html; do
    cp "$f" "$CONFINE_REAL/$(basename "$f")"
done

# Scope the stage to ONLY our published tree.
git add -- "$DST_REL"

if git diff --cached --quiet -- "$DST_REL"; then
    git reset --quiet HEAD -- "$DST_REL" >/dev/null 2>&1 || true
    AHEAD="$(git rev-list --count '@{u}..HEAD' 2>/dev/null || echo 0)"
    if [ "$AHEAD" -gt 0 ]; then
        log "no-op render, but $AHEAD local commit(s) ahead of upstream — pushing"
        git push --quiet
        log "done (recovered unpushed commits)"
        exit 0
    fi
    log "no-op: rendered docs match the published copy (no commit)"
    exit 0
fi

DATE_TAG="$(date -u +%Y-%m-%d)"
MSG="docs: $DATE_TAG publish"
log "committing: $MSG"
git commit -m "$MSG" --quiet -- "$DST_REL"

log "pushing to origin"
git push --quiet
log "done"
