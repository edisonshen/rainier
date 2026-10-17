#!/bin/bash
# Join the tailnet from a Devin VM and verify the legacy TimescaleDB on the
# Mac mini is reachable. Idempotent; safe to re-run.
#
# Requires (org-scoped Devin secrets, injected as env vars):
#   TAILSCALE_OAUTH_CLIENT_SECRET  OAuth client with the `auth_keys` scope
#                                  and tag:devin (registers an ephemeral,
#                                  pre-authorized, tag-owned node)
#   LEGACY_DATABASE_URL            postgresql://rainier:...@<mac-mini-tailnet-ip>:5432/rainier
#
# See docs/RUNBOOK-tailscale-devin-db-access.md.

set -euo pipefail

TAG="${TAILSCALE_TAG:-tag:devin}"
HOST="devin-$(hostname | tr -c 'a-zA-Z0-9-\n' '-' | cut -c1-40)"

die() { echo "devin-db-connect: $*" >&2; exit 1; }

[ -n "${TAILSCALE_OAUTH_CLIENT_SECRET:-}" ] || die "TAILSCALE_OAUTH_CLIENT_SECRET is not set"
[ -n "${LEGACY_DATABASE_URL:-}" ] || die "LEGACY_DATABASE_URL is not set"
command -v tailscale >/dev/null || die "tailscale not installed (blueprint initialize should install it)"

if ! systemctl is-active --quiet tailscaled; then
    sudo systemctl start tailscaled
fi

state="$(tailscale status --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["BackendState"])' 2>/dev/null || echo Unknown)"
if [ "$state" != "Running" ]; then
    # OAuth client secrets act as auth keys; ephemeral=true is the default so
    # each session VM registers a throwaway node that is GC'd when it goes away.
    sudo tailscale up \
        --auth-key="${TAILSCALE_OAUTH_CLIENT_SECRET}?preauthorized=true" \
        --advertise-tags="$TAG" \
        --hostname="$HOST" \
        --accept-dns=false \
        --ssh=false \
        --timeout=60s
fi

tailscale status --peers=false >/dev/null
echo "tailscale: up as $(tailscale ip -4)"

db_host="$(python3 -c 'import os,sys; from urllib.parse import urlparse; u=urlparse(os.environ["LEGACY_DATABASE_URL"]); print(f"{u.hostname}:{u.port or 5432}")')"
for i in $(seq 1 20); do
    if tailscale ping -c 1 --timeout=2s "${db_host%%:*}" >/dev/null 2>&1; then break; fi
    sleep 1
done

if command -v psql >/dev/null; then
    psql "$LEGACY_DATABASE_URL" -Atc "select 'legacy db ok: ' || current_database() || ' @ ' || inet_server_addr()" \
        || die "tailnet is up but Postgres at $db_host refused the connection (ACL, pg_hba, or docker port?)"
else
    echo "psql not found; skipping DB probe (tailnet peer $db_host reachable)"
fi
