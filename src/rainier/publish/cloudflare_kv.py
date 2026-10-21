"""Write one value to a Cloudflare Workers KV namespace via the REST API."""

from __future__ import annotations

from urllib.parse import quote

import httpx

API = "https://api.cloudflare.com/client/v4"


def put_kv(account_id: str, namespace_id: str, key: str, value: str, token: str) -> None:
    url = f"{API}/accounts/{account_id}/storage/kv/namespaces/{namespace_id}/values/{quote(key, safe='')}"
    resp = httpx.put(
        url, content=value.encode(), timeout=30,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "text/plain"},
    )
    resp.raise_for_status()
