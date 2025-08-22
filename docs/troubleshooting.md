# Troubleshooting

## UI/Queue errors (fn_index KeyError)
- After code changes, do a hard refresh (Ctrl+F5) or restart the server. Stale frontends can reference removed event handlers.

## Tool output parsing
- `web_search` returns plain text. Do not `json.loads` or index it.
- `hf_*` tools return JSON as strings. Always `json.loads` before indexing.

## Rate limits
- DuckDuckGo/site search can rate limit. Reduce `limit`, vary queries, retry later.

## Windows Unicode
- If console prints error on Unicode, ensure UTF‑8 code page or encode output explicitly.

## Access errors
- 401/403 is expected for gated/private. Tools should mark `access=no_access`.
