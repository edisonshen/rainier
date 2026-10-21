"""Write one value to a Cloudflare Workers KV namespace via the REST API."""

from __future__ import annotations

from urllib.parse import quote

import httpx

API = "https://api.cloudflare.com/client/v4"


def _url(account_id: str, namespace_id: str, key: str) -> str:
    return f"{API}/accounts/{account_id}/storage/kv/namespaces/{namespace_id}/values/{quote(key, safe='')}"


def get_kv(account_id: str, namespace_id: str, key: str, token: str) -> str | None:
    resp = httpx.get(_url(account_id, namespace_id, key), timeout=30,
                     headers={"Authorization": f"Bearer {token}"})
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.text


def put_kv(account_id: str, namespace_id: str, key: str, value: str, token: str) -> None:
    resp = httpx.put(
        _url(account_id, namespace_id, key), content=value.encode(), timeout=30,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "text/plain"},
    )
    resp.raise_for_status()


def put_kv_if_changed(account_id: str, namespace_id: str, key: str, value: str,
                      token: str) -> bool:
    """Write only when ``value`` differs from what KV holds; returns whether it wrote."""
    if get_kv(account_id, namespace_id, key, token) == value:
        return False
    put_kv(account_id, namespace_id, key, value, token)
    return True
