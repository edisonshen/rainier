# RUNBOOK — Devin access to the Mac mini TimescaleDB over Tailscale

Goal: let Devin cloud sessions (and any other tailnet machine) reach the legacy
`public.*` TimescaleDB that runs in docker on the Mac mini, without moving the
data to a cloud database and without exposing Postgres to the internet.

```
Devin VM (ephemeral tag:devin node)          Mac mini (tag:rainier-db)
┌──────────────────────────────┐   WireGuard   ┌──────────────────────────────┐
│ scripts/devin-db-connect.sh  │ ───────────▶  │ tailscaled                   │
│   tailscale up (OAuth key)   │  100.x.y.z    │ docker: timescaledb :5432    │
│   psql $LEGACY_DATABASE_URL  │   :5432       │ (docker-compose.yaml)        │
└──────────────────────────────┘               └──────────────────────────────┘
        ACL: tag:devin → tag:rainier-db:5432 only
```

Cost: $0 (Tailscale Personal plan). Data stays on the Mac mini; the nightly
`money-flow-backup` job to Neon remains the off-site copy.

## 1. Tailnet setup (one time, admin console)

1. **Tags + ACL** — Access controls → edit the policy file:

   ```jsonc
   {
     "tagOwners": {
       "tag:devin":      ["autogroup:admin"],
       "tag:rainier-db": ["autogroup:admin"]
     },
     "acls": [
       // keep your existing rules, then:
       { "action": "accept", "src": ["tag:devin"], "dst": ["tag:rainier-db:5432"] }
     ]
   }
   ```

   Devin nodes can reach nothing on the tailnet except Postgres on the DB host.

2. **OAuth client** — Settings → Trust credentials → Generate credential → OAuth.
   Scope: `Auth Keys` (write). Tag: `tag:devin`. Copy the client secret
   (`tskey-client-…`). It is used directly as an auth key
   (`tailscale up --auth-key=$SECRET --advertise-tags=tag:devin`) and, unlike a
   plain auth key, does not expire after 90 days. Nodes it registers are
   ephemeral by default, so each Devin VM is removed from the tailnet
   automatically when the session machine goes away.

## 2. Mac mini setup

```bash
brew install --cask tailscale-app          # or the App Store / tailscale.com package
tailscale up --advertise-tags=tag:rainier-db
tailscale ip -4                            # → 100.x.y.z, stable for this node
```

Postgres is already published on all interfaces by `docker-compose.yaml`
(`"5432:5432"`), so the tailnet IP reaches it with no docker changes. If the
macOS firewall is on, allow incoming connections for Docker.

**Rotate the DB password.** `rainier_dev` was fine for localhost-only; it is
now reachable by any node the ACL admits. In `docker-compose.yaml` set
`POSTGRES_PASSWORD` to a real secret, `docker compose up -d`, and update
`LEGACY_DATABASE_URL` in the Mac mini `.env` (the password in the compose file
only takes effect on a fresh volume — for an existing volume run
`ALTER USER rainier PASSWORD '...'` via `docker compose exec db psql -U rainier`).

## 3. Devin setup

Org-scoped secrets (Settings → Secrets), both visible to snapshot builds and
injected into every session shell:

| Secret | Value |
|---|---|
| `TAILSCALE_OAUTH_CLIENT_SECRET` | the `tskey-client-…` from §1.2 |
| `LEGACY_DATABASE_URL` | `postgresql://rainier:<password>@100.x.y.z:5432/rainier` |

Use the tailnet IP rather than the MagicDNS name: the connect script runs
`tailscale up --accept-dns=false` so it never rewrites the VM's resolver.

The rainier blueprint installs `tailscale` (and the Postgres client) into the
snapshot. In a session, run:

```bash
scripts/devin-db-connect.sh
```

It starts `tailscaled` if needed, joins the tailnet as an ephemeral
`devin-<host>` node, waits for the DB peer, and runs a `SELECT` through
`LEGACY_DATABASE_URL`. After that, `uv run rainier …` commands and `psql
"$LEGACY_DATABASE_URL"` talk to the Mac mini directly.

## 4. Verify / troubleshoot

| Symptom | Check |
|---|---|
| `tailscale up` hangs / "key not authorized" | OAuth client lacks the `auth_keys` scope or `tag:devin`; `tag:devin` missing from `tagOwners` |
| tailnet up, `psql` times out | ACL missing `tag:devin → tag:rainier-db:5432`; Mac mini not tagged `tag:rainier-db`; Mac mini asleep (System Settings → Energy → prevent sleep) |
| `psql` → "no pg_hba.conf entry" | Docker image default is `host all all all scram-sha-256`; only fires if you customised `pg_hba.conf` |
| Password auth failed | Password rotated in compose but not applied to the existing volume — `ALTER USER` (§2) |

Devin-side: `tailscale status`, `tailscale ping <db-ip>`, `tailscale netcheck`.
Admin console → Machines shows `devin-*` ephemeral nodes while sessions are live.

## 5. Related

- `src/rainier/db/money_flow_backup.py` + `config/cron.yaml` `money-flow-backup` — off-site copy of `money_flow_snapshots` on Neon (unchanged by this runbook).
- `.env.example` — `LEGACY_DATABASE_URL` vs `DATABASE_URL` split.
