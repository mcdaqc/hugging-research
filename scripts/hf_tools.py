import json
import time
from typing import Dict, List, Optional, Tuple

import requests
from smolagents import Tool


# -----------------------------
# HTTP helpers (anonymous only)
# -----------------------------

DEFAULT_TIMEOUT = 15
RETRY_STATUS = {429, 500, 502, 503, 504}


def _anonymous_headers() -> Dict[str, str]:
    return {
        "Accept": "application/json",
        "User-Agent": "HuggingResearch-Agent/1.0 (+https://huggingface.co)",
        # No Authorization header on purpose (public only)
    }


def _http_get_json(url: str, params: Optional[Dict] = None, max_retries: int = 2) -> Tuple[Optional[Dict | List], int, str]:
    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params or {}, headers=_anonymous_headers(), timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 200:
                try:
                    return resp.json(), resp.status_code, ""
                except Exception as je:
                    return None, resp.status_code, f"invalid_json: {je}"
            if resp.status_code in {401, 403}:
                # Private/Gated/Unauthorized
                return None, resp.status_code, "no_access"
            if resp.status_code in RETRY_STATUS and attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            return None, resp.status_code, f"http_{resp.status_code}"
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                time.sleep(0.8 * (attempt + 1))
                continue
            return None, 0, f"exception: {last_err}"


# -----------------------------
# Normalization helpers
# -----------------------------

def _visibility_from_item(item: Dict) -> Tuple[str, str]:
    if not isinstance(item, dict):
        return "public", "accessible"
    if item.get("private") is True:
        return "private", "no_access"
    if item.get("gated") is True or item.get("gatedReason") or (isinstance(item.get("cardData"), dict) and item["cardData"].get("gated")):
        return "gated", "no_access"
    return "public", "accessible"


def _norm_common(item_id: str, item_type: str, owner: str, description: str = "", url_suffix: str = "") -> Dict:
    url = f"https://huggingface.co/{url_suffix}{item_id}" if url_suffix else f"https://huggingface.co/{item_id}"
    return {
        "type": item_type,
        "id": item_id,
        "owner": owner,
        "url": url,
        "description": description or "",
    }


def _safe_get(item: Dict, key: str, default=None):
    return item.get(key, default) if isinstance(item, dict) else default


# -----------------------------
# Tools
# -----------------------------


class HFModelsSearchTool(Tool):
    name = "hf_models_search"
    description = (
        "Search public Hugging Face models. Provide a free-text query and optional filters "
        "(owner, single pipeline_tag, tags CSV, sort/direction, limit). "
        "Prefer minimal params; add owner/task/tags/sort only when the user implies them. "
        "Defaults: limit=10, sort omitted, direction omitted. Returns JSON with `results`, `status`, `error`, and `params`."
    )
    inputs = {
        "query": {"type": "string", "description": "Free-text search", "nullable": True},
        "owner": {"type": "string", "description": "Filter by owner/namespace", "nullable": True},
        "task": {"type": "string", "description": "Primary pipeline tag, e.g. text-classification", "nullable": True},
        "tags": {"type": "string", "description": "Comma-separated tags filter", "nullable": True},
        "sort": {"type": "string", "description": "downloads|likes|modified", "nullable": True},
        "direction": {"type": "string", "description": "descending|ascending", "nullable": True},
        "limit": {"type": "number", "description": "Max results", "nullable": True},
    }
    output_type = "string"

    def forward(self, query: Optional[str] = None, owner: Optional[str] = None, task: Optional[str] = None, tags: Optional[str] = None, sort: Optional[str] = None, direction: Optional[str] = None, limit: Optional[int] = None) -> str:
        # Build conservative params
        params = {}
        if query:
            params["search"] = query
        if owner:
            params["author"] = owner
        if task:
            # pipeline_tag must be a single value; if multiple provided, take the first
            first_task = task.split(",")[0].strip()
            if first_task:
                params["pipeline_tag"] = first_task
        if tags:
            # Support comma-separated → repeated tags
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]
            if len(tag_list) == 1:
                params["tags"] = tag_list[0]
            elif len(tag_list) > 1:
                params["tags"] = tag_list  # requests will repeat param
        # Support 'trending' as an alias mapped to downloads+descending for recency/interest
        if sort in {"downloads", "likes", "modified", "trending"}:
            params["sort"] = sort
        if sort == "trending":
            params["sort"] = "downloads"
            params["direction"] = "descending"
        elif direction in {"descending", "ascending"}:
            params["direction"] = direction
        # Default limit to 10 if not specified
        lim = int(limit) if limit else 10
        params["limit"] = lim

        data, status, err = _http_get_json("https://huggingface.co/api/models", params)
        # Fallback: retry with minimal params if 400
        if status == 400:
            minimal = {"search": query} if query else {}
            if limit:
                minimal["limit"] = int(limit)
            data, status, err = _http_get_json("https://huggingface.co/api/models", minimal)
        results: List[Dict] = []
        if isinstance(data, list):
            for it in data:
                model_id = _safe_get(it, "id") or _safe_get(it, "modelId") or ""
                if not model_id:
                    continue
                owner_name = model_id.split("/")[0] if "/" in model_id else ""
                desc = ""
                # If present, short description may live in cardData/summary when full=true; not guaranteed in list
                visibility, access = _visibility_from_item(it)
                norm = _norm_common(model_id, "model", owner_name, desc)
                norm.update({
                    "tags": _safe_get(it, "tags", []),
                    "task": _safe_get(it, "pipeline_tag"),
                    "likes": _safe_get(it, "likes", 0),
                    "downloads": _safe_get(it, "downloads", 0),
                    "updatedAt": _safe_get(it, "lastModified"),
                    "visibility": visibility,
                    "access": access,
                })
                results.append(norm)

        return json.dumps({
            "results": results,
            "status": status,
            "error": err,
            "params": params,
        }, ensure_ascii=False)


class HFModelInfoTool(Tool):
    name = "hf_model_info"
    description = (
        "Get detailed public model info by repo id (owner/name). Use this after a model search to fetch richer metadata (cardData, siblings, tags)."
    )
    inputs = {
        "repo_id": {"type": "string", "description": "Model repo id, e.g. bigscience/bloom"}
    }
    output_type = "string"

    def forward(self, repo_id: str) -> str:
        data, status, err = _http_get_json(f"https://huggingface.co/api/models/{repo_id}", {"full": "true"})
        item: Dict = {}
        if isinstance(data, dict):
            model_id = data.get("id") or data.get("modelId") or repo_id
            owner_name = model_id.split("/")[0] if "/" in model_id else ""
            visibility, access = _visibility_from_item(data)
            desc = ""
            # Some cards put a short summary in cardData/summary
            if isinstance(data.get("cardData"), dict):
                desc = data["cardData"].get("summary") or data["cardData"].get("description") or ""
            item = _norm_common(model_id, "model", owner_name, desc)
            item.update({
                "tags": data.get("tags", []),
                "task": data.get("pipeline_tag"),
                "likes": data.get("likes", 0),
                "downloads": data.get("downloads", 0),
                "updatedAt": data.get("lastModified"),
                "visibility": visibility,
                "access": access,
                "cardData": data.get("cardData"),
                "siblings": data.get("siblings"),
            })
        return json.dumps({"item": item, "status": status, "error": err}, ensure_ascii=False)


class HFDatasetsSearchTool(Tool):
    name = "hf_datasets_search"
    description = (
        "Search public datasets with a free-text query and optional filters (owner, tags CSV, sort/direction, limit). "
        "Prefer minimal params; add filters when implied. Defaults: limit=10. Returns JSON with `results`, `status`, `error`, and `params`."
    )
    inputs = {
        "query": {"type": "string", "description": "Free-text search", "nullable": True},
        "owner": {"type": "string", "description": "Filter by owner/namespace", "nullable": True},
        "tags": {"type": "string", "description": "Comma-separated tags filter", "nullable": True},
        "sort": {"type": "string", "description": "downloads|likes|modified", "nullable": True},
        "direction": {"type": "string", "description": "descending|ascending", "nullable": True},
        "limit": {"type": "number", "description": "Max results", "nullable": True},
    }
    output_type = "string"

    def forward(self, query: Optional[str] = None, owner: Optional[str] = None, tags: Optional[str] = None, sort: Optional[str] = None, direction: Optional[str] = None, limit: Optional[int] = None) -> str:
        params = {}
        if query:
            params["search"] = query
        if owner:
            params["author"] = owner
        if tags:
            tag_list = [t.strip() for t in tags.split(",")] if isinstance(tags, str) else []
            tag_list = [t for t in tag_list if t]
            if len(tag_list) == 1:
                params["tags"] = tag_list[0]
            elif len(tag_list) > 1:
                params["tags"] = tag_list
        if sort in {"downloads", "likes", "modified", "trending"}:
            params["sort"] = sort
        if sort == "trending":
            params["sort"] = "downloads"
            params["direction"] = "descending"
        elif direction in {"descending", "ascending"}:
            params["direction"] = direction
        lim = int(limit) if limit else 10
        params["limit"] = lim

        data, status, err = _http_get_json("https://huggingface.co/api/datasets", params)
        if status == 400:
            minimal = {"search": query} if query else {}
            if limit:
                minimal["limit"] = int(limit)
            data, status, err = _http_get_json("https://huggingface.co/api/datasets", minimal)
        results: List[Dict] = []
        if isinstance(data, list):
            for it in data:
                ds_id = _safe_get(it, "id") or _safe_get(it, "datasetId") or ""
                if not ds_id:
                    continue
                owner_name = ds_id.split("/")[0] if "/" in ds_id else ""
                visibility, access = _visibility_from_item(it)
                norm = _norm_common(ds_id, "dataset", owner_name, "")
                norm.update({
                    "tags": _safe_get(it, "tags", []),
                    "likes": _safe_get(it, "likes", 0),
                    "downloads": _safe_get(it, "downloads", 0),
                    "updatedAt": _safe_get(it, "lastModified"),
                    "visibility": visibility,
                    "access": access,
                })
                results.append(norm)
        return json.dumps({"results": results, "status": status, "error": err, "params": params}, ensure_ascii=False)


class HFDatasetInfoTool(Tool):
    name = "hf_dataset_info"
    description = (
        "Get detailed public dataset info by repo id (owner/name). Use after a dataset search to retrieve cardData and siblings."
    )
    inputs = {"repo_id": {"type": "string", "description": "Dataset repo id, e.g. glue"}}
    output_type = "string"

    def forward(self, repo_id: str) -> str:
        data, status, err = _http_get_json(f"https://huggingface.co/api/datasets/{repo_id}", {"full": "true"})
        item: Dict = {}
        if isinstance(data, dict):
            ds_id = data.get("id") or data.get("datasetId") or repo_id
            owner_name = ds_id.split("/")[0] if "/" in ds_id else ""
            visibility, access = _visibility_from_item(data)
            desc = ""
            if isinstance(data.get("cardData"), dict):
                desc = data["cardData"].get("summary") or data["cardData"].get("description") or ""
            item = _norm_common(ds_id, "dataset", owner_name, desc)
            item.update({
                "tags": data.get("tags", []),
                "likes": data.get("likes", 0),
                "downloads": data.get("downloads", 0),
                "updatedAt": data.get("lastModified"),
                "visibility": visibility,
                "access": access,
                "cardData": data.get("cardData"),
                "siblings": data.get("siblings"),
            })
        return json.dumps({"item": item, "status": status, "error": err}, ensure_ascii=False)


class HFSpacesSearchTool(Tool):
    name = "hf_spaces_search"
    description = (
        "Search public Spaces with query and optional filters (owner, tags CSV, sort/direction, limit). "
        "Good for tutorials/demos related to a topic. Defaults: limit=10. Returns JSON with `results`, `status`, `error`, and `params`."
    )
    inputs = {
        "query": {"type": "string", "description": "Free-text search", "nullable": True},
        "owner": {"type": "string", "description": "Filter by owner/namespace", "nullable": True},
        "tags": {"type": "string", "description": "Comma-separated tags filter", "nullable": True},
        "sort": {"type": "string", "description": "likes|modified", "nullable": True},
        "direction": {"type": "string", "description": "descending|ascending", "nullable": True},
        "limit": {"type": "number", "description": "Max results", "nullable": True},
    }
    output_type = "string"

    def forward(self, query: Optional[str] = None, owner: Optional[str] = None, tags: Optional[str] = None, sort: Optional[str] = None, direction: Optional[str] = None, limit: Optional[int] = None) -> str:
        params = {}
        if query:
            params["search"] = query
        if owner:
            params["author"] = owner
        if tags:
            tag_list = [t.strip() for t in tags.split(",")] if isinstance(tags, str) else []
            tag_list = [t for t in tag_list if t]
            if len(tag_list) == 1:
                params["tags"] = tag_list[0]
            elif len(tag_list) > 1:
                params["tags"] = tag_list
        if sort in {"likes", "modified", "trending"}:
            params["sort"] = sort
        if sort == "trending":
            params["sort"] = "likes"
            params["direction"] = "descending"
        elif direction in {"descending", "ascending"}:
            params["direction"] = direction
        lim = int(limit) if limit else 10
        params["limit"] = lim

        data, status, err = _http_get_json("https://huggingface.co/api/spaces", params)
        if status == 400:
            minimal = {"search": query} if query else {}
            if limit:
                minimal["limit"] = int(limit)
            data, status, err = _http_get_json("https://huggingface.co/api/spaces", minimal)
        results: List[Dict] = []
        if isinstance(data, list):
            for it in data:
                sp_id = _safe_get(it, "id") or _safe_get(it, "spaceId") or ""
                if not sp_id:
                    continue
                owner_name = sp_id.split("/")[0] if "/" in sp_id else ""
                visibility, access = _visibility_from_item(it)
                norm = _norm_common(sp_id, "space", owner_name, "")
                # Try to extract Space runtime (sdk, app file) when available in list
                norm.update({
                    "tags": _safe_get(it, "tags", []),
                    "likes": _safe_get(it, "likes", 0),
                    "downloads": _safe_get(it, "downloads", 0),
                    "updatedAt": _safe_get(it, "lastModified"),
                    "visibility": visibility,
                    "access": access,
                })
                results.append(norm)
        return json.dumps({"results": results, "status": status, "error": err, "params": params}, ensure_ascii=False)


class HFSpaceInfoTool(Tool):
    name = "hf_space_info"
    description = (
        "Get detailed Space info by repo id (owner/name). Use to inspect tags, likes, and card details after a Space search."
    )
    inputs = {"repo_id": {"type": "string", "description": "Space repo id, e.g. user/space-name"}}
    output_type = "string"

    def forward(self, repo_id: str) -> str:
        data, status, err = _http_get_json(f"https://huggingface.co/api/spaces/{repo_id}", {"full": "true"})
        item: Dict = {}
        if isinstance(data, dict):
            sp_id = data.get("id") or data.get("spaceId") or repo_id
            owner_name = sp_id.split("/")[0] if "/" in sp_id else ""
            visibility, access = _visibility_from_item(data)
            desc = ""
            if isinstance(data.get("cardData"), dict):
                desc = data["cardData"].get("summary") or data["cardData"].get("description") or ""
            item = _norm_common(sp_id, "space", owner_name, desc)
            item.update({
                "tags": data.get("tags", []),
                "likes": data.get("likes", 0),
                "downloads": data.get("downloads", 0),
                "updatedAt": data.get("lastModified"),
                "visibility": visibility,
                "access": access,
                "cardData": data.get("cardData"),
                "siblings": data.get("siblings"),
            })
        return json.dumps({"item": item, "status": status, "error": err}, ensure_ascii=False)


class HFUserInfoTool(Tool):
    name = "hf_user_info"
    description = (
        "Fetch public user/org profile by username. Helpful to scope searches by owner or explore maintainers."
    )
    inputs = {"username": {"type": "string", "description": "User or organization name"}}
    output_type = "string"

    def forward(self, username: str) -> str:
        data, status, err = _http_get_json(f"https://huggingface.co/api/users/{username}")
        item = data if isinstance(data, dict) else {}
        visibility = "public"
        access = "accessible" if status == 200 else "no_access"
        return json.dumps({"item": item, "status": status, "error": err, "visibility": visibility, "access": access}, ensure_ascii=False)


class HFCollectionsListTool(Tool):
    name = "hf_collections_list"
    description = (
        "List public collections, optionally filtered by owner/namespace. Use to surface curated sets of repos. "
        "Owner may be an object; URL is normalized to https://huggingface.co/collections/{owner_name}/{slug}."
    )
    inputs = {"owner": {"type": "string", "description": "Filter by collection owner/namespace", "nullable": True}}
    output_type = "string"

    def forward(self, owner: Optional[str] = None) -> str:
        params = {}
        if owner:
            params["owner"] = owner
        data, status, err = _http_get_json("https://huggingface.co/api/collections", params)
        results = data if isinstance(data, list) else []
        # Normalize minimally
        items: List[Dict] = []
        for it in results:
            cid = _safe_get(it, "id") or _safe_get(it, "slug") or ""
            ns_val = _safe_get(it, "owner") or _safe_get(it, "namespace") or ""
            if isinstance(ns_val, dict):
                ns = ns_val.get("name") or ns_val.get("fullname") or ""
            else:
                ns = ns_val
            url = ""
            if ns and cid:
                # Some APIs return id as "{namespace}/{slug}", so extract slug part only
                slug = cid.split("/")[-1]
                url = f"https://huggingface.co/collections/{ns}/{slug}"
            items.append({
                "type": "collection",
                "id": cid,
                "owner": ns,
                "title": _safe_get(it, "title", ""),
                "url": url,
                "visibility": "public",
                "access": "accessible",
            })
        return json.dumps({"results": items, "status": status, "error": err}, ensure_ascii=False)


class HFCollectionGetTool(Tool):
    name = "hf_collection_get"
    description = (
        "Get collection details by namespace and slug id (as in URL). Use after listing to inspect items."
    )
    inputs = {
        "namespace": {"type": "string", "description": "Collection owner/namespace"},
        "slug_id": {"type": "string", "description": "slug-id part as shown in URL"},
    }
    output_type = "string"

    def forward(self, namespace: str, slug_id: str) -> str:
        data, status, err = _http_get_json(f"https://huggingface.co/api/collections/{namespace}/{slug_id}")
        item = data if isinstance(data, dict) else {}
        return json.dumps({"item": item, "status": status, "error": err}, ensure_ascii=False)


class HFPaperInfoTool(Tool):
    name = "hf_paper_info"
    description = (
        "Fetch paper metadata by arXiv id (e.g., 1706.03762). Combine with hf_paper_repos to find related repos."
    )
    inputs = {"arxiv_id": {"type": "string", "description": "arXiv identifier, e.g. 1706.03762"}}
    output_type = "string"

    def forward(self, arxiv_id: str) -> str:
        data, status, err = _http_get_json(f"https://huggingface.co/api/papers/{arxiv_id}")
        item = data if isinstance(data, dict) else {}
        return json.dumps({"item": item, "status": status, "error": err}, ensure_ascii=False)


class HFPaperReposTool(Tool):
    name = "hf_paper_repos"
    description = (
        "List repos (models/datasets/spaces) referencing an arXiv id. Use alongside hf_paper_info to map research → repos."
    )
    inputs = {"arxiv_id": {"type": "string", "description": "arXiv identifier, e.g. 1706.03762"}}
    output_type = "string"

    def forward(self, arxiv_id: str) -> str:
        data, status, err = _http_get_json(f"https://huggingface.co/api/arxiv/{arxiv_id}/repos")
        results = data if isinstance(data, list) else []
        return json.dumps({"results": results, "status": status, "error": err}, ensure_ascii=False)


class HFDailyPapersTool(Tool):
    name = "hf_daily_papers"
    description = (
        "Get the daily curated papers list from Hugging Face. Useful for current research trends."
    )
    inputs = {}
    output_type = "string"

    def forward(self) -> str:  # type: ignore[override]
        data, status, err = _http_get_json("https://huggingface.co/api/daily_papers")
        results = data if isinstance(data, list) else []
        return json.dumps({"results": results, "status": status, "error": err}, ensure_ascii=False)


class HFRepoInfoTool(Tool):
    name = "hf_repo_info"
    description = (
        "Generic repo info for model|dataset|space by id. Use if you already know the type and want raw item metadata."
    )
    inputs = {
        "repo_type": {"type": "string", "description": "model|dataset|space"},
        "repo_id": {"type": "string", "description": "Owner/name or id"},
    }
    output_type = "string"

    def forward(self, repo_type: str, repo_id: str) -> str:
        repo_type = (repo_type or "").strip().lower()
        if repo_type not in {"model", "dataset", "space"}:
            return json.dumps({"error": "invalid_repo_type", "status": 400})
        base = {"model": "models", "dataset": "datasets", "space": "spaces"}[repo_type]
        data, status, err = _http_get_json(f"https://huggingface.co/api/{base}/{repo_id}", {"full": "true"})
        item = data if isinstance(data, dict) else {}
        return json.dumps({"item": item, "status": status, "error": err}, ensure_ascii=False)


class HFSiteSearchTool(Tool):
    name = "hf_site_search"
    description = (
        "Search within huggingface.co for blogs, Learn pages, and posts (DuckDuckGo). Prefer this for tutorials and docs not covered by Hub APIs. "
        "Defaults: limit=10 to reduce rate limiting. Returns JSON with `results`, `status`, and `error`."
    )
    inputs = {
        "query": {"type": "string", "description": "Search query. 'site:huggingface.co' will be added if missing."},
        "limit": {"type": "number", "description": "Max results (default 20)", "nullable": True},
    }
    output_type = "string"

    def forward(self, query: str, limit: Optional[int] = None) -> str:
        try:
            from duckduckgo_search import DDGS
        except Exception:
            return json.dumps({"results": [], "status": 500, "error": "duckduckgo_search_not_installed"})

        q = f"site:huggingface.co {query}" if "huggingface.co" not in query else query
        lim = int(limit) if limit else 10
        results: List[Dict] = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(q, safesearch="moderate", timelimit=None, max_results=lim):
                    if not isinstance(r, dict):
                        continue
                    results.append({
                        "type": "site",
                        "title": r.get("title"),
                        "url": r.get("href"),
                        "snippet": r.get("body"),
                        "date": r.get("date"),
                    })
        except Exception as e:
            return json.dumps({"results": [], "status": 500, "error": str(e)})
        return json.dumps({"results": results, "status": 200, "error": ""}, ensure_ascii=False)


class HFReportGenerateTool(Tool):
    name = "hf_report_generate"
    description = (
        "Generate a full HTML report from aggregated JSON (string). The app prefers its own dashboard, but this can render custom summaries."
    )
    inputs = {
        "data_json": {"type": "string", "description": "Aggregated search results JSON"},
        "title": {"type": "string", "description": "Report title", "nullable": True},
    }
    output_type = "string"

    def forward(self, data_json: str, title: Optional[str] = None) -> str:
        try:
            data = json.loads(data_json) if data_json else {}
        except Exception as e:
            data = {"parse_error": str(e)}
        title = title or "Hugging Face Research Report"

        def card_html(item: Dict) -> str:
            badge = ""
            vis = item.get("visibility")
            access = item.get("access")
            if vis in {"private", "gated"} or access == "no_access":
                badge = f"<span class=badge badge-warn>{vis or 'restricted'}</span>"
            meta = []
            if item.get("task"):
                meta.append(f"<span class=meta>Task: {item['task']}</span>")
            if item.get("tags"):
                meta.append(f"<span class=meta>Tags: {', '.join(item['tags'][:5])}</span>")
            if item.get("downloads") is not None:
                meta.append(f"<span class=stat>⬇️ {item['downloads']}</span>")
            if item.get("likes") is not None:
                meta.append(f"<span class=stat>❤️ {item['likes']}</span>")
            if item.get("updatedAt"):
                meta.append(f"<span class=meta>Updated: {item['updatedAt']}</span>")
            desc = (item.get("description") or "").strip()
            if len(desc) > 220:
                desc = desc[:217] + "..."
            return (
                "<div class=card>"
                f"<div class=card-title><a href='{item.get('url')}' target=_blank rel=noopener>{item.get('id')}</a> {badge}</div>"
                f"<div class=card-subtitle>{item.get('type','')} • {item.get('owner','')}</div>"
                f"<div class=card-desc>{desc}</div>"
                f"<div class=card-meta>{' | '.join(meta)}</div>"
                "</div>"
            )

        def section(title_text: str, items: List[Dict]) -> str:
            if not items:
                return ""
            cards = "\n".join(card_html(it) for it in items)
            return f"<section><h2>{title_text}</h2><div class=cards>{cards}</div></section>"

        # Accept either a dict with category keys or a flat list
        models = data.get("models") or data.get("Models") or []
        datasets = data.get("datasets") or data.get("Datasets") or []
        spaces = data.get("spaces") or data.get("Spaces") or []
        papers = data.get("papers") or data.get("Papers") or []
        daily_papers = data.get("daily_papers") or data.get("DailyPapers") or []
        users = data.get("users") or data.get("Users") or []
        collections = data.get("collections") or data.get("Collections") or []
        site = data.get("site") or data.get("Site") or []

        html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <style>
    :root {{ --bg:#0b0d12; --fg:#e6e9ef; --muted:#9aa4b2; --card:#121621; --accent:#5ac8fa; --warn:#eab308; }}
    body {{ background:var(--bg); color:var(--fg); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Inter, Arial, sans-serif; margin:0; padding:24px; }}
    h1 {{ font-size: 24px; margin: 0 0 12px; }}
    h2 {{ font-size: 18px; margin: 24px 0 8px; color: var(--accent); }}
    .container {{ max-width: 1120px; margin: 0 auto; }}
    .subtitle {{ color: var(--muted); margin-bottom: 18px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px,1fr)); gap: 12px; }}
    .card {{ background: var(--card); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 12px; }}
    .card-title {{ font-weight: 600; margin-bottom: 4px; overflow-wrap:anywhere; }}
    .card-subtitle {{ color: var(--muted); font-size: 12px; margin-bottom: 8px; }}
    .card-desc {{ font-size: 13px; line-height: 1.45; min-height: 28px; margin-bottom: 8px; color: #d2d7df; }}
    .card-meta {{ font-size: 12px; color: var(--muted); display:flex; flex-wrap:wrap; gap:8px; }}
    .badge {{ background: rgba(234, 179, 8, 0.15); color: #facc15; border:1px solid rgba(250,204,21,0.35); border-radius: 999px; padding: 2px 8px; font-size: 11px; margin-left: 6px; }}
    .badge-warn {{ background: rgba(234, 179, 8, 0.15); }}
    a {{ color: #93c5fd; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    section {{ margin-bottom: 18px; }}
  </style>
  <script>
    function printToPDF() {{ window.print(); }}
  </script>
  <link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/modern-normalize/2.0.0/modern-normalize.min.css\" />
  <meta name=\"robots\" content=\"noindex\" />
  <meta name=\"referrer\" content=\"no-referrer\" />
  <meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'self' 'unsafe-inline' data: https://cdnjs.cloudflare.com; img-src * data:; style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com;\" />
</head>
<body>
  <div class=\"container\">
    <div style=\"display:flex; align-items:center; justify-content:space-between; gap:12px;\">
      <div>
        <h1>{title}</h1>
        <div class=\"subtitle\">Generated by Hugging Search</div>
      </div>
      <button onclick=\"printToPDF()\" style=\"background:#1f2937;color:#e5e7eb;border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:8px 10px;cursor:pointer;\">Print to PDF</button>
    </div>
    {section("Models", models)}
    {section("Datasets", datasets)}
    {section("Spaces", spaces)}
    {section("Papers", papers)}
    {section("Daily Papers", daily_papers)}
    {section("Users", users)}
    {section("Collections", collections)}
    {section("Site results", site)}
  </div>
</body>
</html>
"""
        return html


class HFDashboardReportTool(Tool):
    name = "hf_generate_dashboard_report"
    description = (
        "One-click dashboard report from a query. Fetches public models/datasets/spaces/daily_papers and returns a full HTML dashboard."
    )
    inputs = {
        "query": {"type": "string", "description": "User intent / keywords to search across Hub"},
        "limit": {"type": "number", "description": "Max results per category (default 20)", "nullable": True},
    }
    output_type = "string"

    def forward(self, query: str, limit: Optional[int] = None) -> str:
        lim = int(limit) if limit else 20
        params_common = {"search": query, "sort": "downloads", "direction": "descending", "limit": lim}
        # Fetch categories
        m_data, m_status, _ = _http_get_json("https://huggingface.co/api/models", params_common)
        d_data, d_status, _ = _http_get_json("https://huggingface.co/api/datasets", params_common)
        s_data, s_status, _ = _http_get_json("https://huggingface.co/api/spaces", {"search": query, "sort": "likes", "direction": "descending", "limit": lim})
        dp_data, dp_status, _ = _http_get_json("https://huggingface.co/api/daily_papers")

        models: List[Dict] = []
        if isinstance(m_data, list):
            for it in m_data[:lim]:
                model_id = _safe_get(it, "id") or _safe_get(it, "modelId") or ""
                if not model_id:
                    continue
                owner_name = model_id.split("/")[0] if "/" in model_id else ""
                visibility, access = _visibility_from_item(it)
                norm = _norm_common(model_id, "model", owner_name, "")
                norm.update({
                    "tags": _safe_get(it, "tags", []),
                    "task": _safe_get(it, "pipeline_tag"),
                    "likes": _safe_get(it, "likes", 0),
                    "downloads": _safe_get(it, "downloads", 0),
                    "updatedAt": _safe_get(it, "lastModified"),
                    "visibility": visibility,
                    "access": access,
                })
                models.append(norm)

        datasets: List[Dict] = []
        if isinstance(d_data, list):
            for it in d_data[:lim]:
                ds_id = _safe_get(it, "id") or _safe_get(it, "datasetId") or ""
                if not ds_id:
                    continue
                owner_name = ds_id.split("/")[0] if "/" in ds_id else ""
                visibility, access = _visibility_from_item(it)
                norm = _norm_common(ds_id, "dataset", owner_name, "")
                norm.update({
                    "tags": _safe_get(it, "tags", []),
                    "likes": _safe_get(it, "likes", 0),
                    "downloads": _safe_get(it, "downloads", 0),
                    "updatedAt": _safe_get(it, "lastModified"),
                    "visibility": visibility,
                    "access": access,
                })
                datasets.append(norm)

        spaces: List[Dict] = []
        if isinstance(s_data, list):
            for it in s_data[:lim]:
                sp_id = _safe_get(it, "id") or _safe_get(it, "spaceId") or ""
                if not sp_id:
                    continue
                owner_name = sp_id.split("/")[0] if "/" in sp_id else ""
                visibility, access = _visibility_from_item(it)
                norm = _norm_common(sp_id, "space", owner_name, "")
                norm.update({
                    "tags": _safe_get(it, "tags", []),
                    "likes": _safe_get(it, "likes", 0),
                    "downloads": _safe_get(it, "downloads", 0),
                    "updatedAt": _safe_get(it, "lastModified"),
                    "visibility": visibility,
                    "access": access,
                })
                spaces.append(norm)

        papers = dp_data if isinstance(dp_data, list) else []

        # Build dashboard HTML
        def card_html(item: Dict) -> str:
            badge = ""
            if item.get("visibility") in {"private", "gated"} or item.get("access") == "no_access":
                badge = f"<span class=badge badge-warn>{item.get('visibility','restricted')}</span>"
            meta = []
            if item.get("task"):
                meta.append(f"<span class=meta>Task: {item['task']}</span>")
            if item.get("tags"):
                meta.append(f"<span class=meta>Tags: {', '.join(item['tags'][:5])}</span>")
            if item.get("downloads") is not None:
                meta.append(f"<span class=stat>⬇️ {item['downloads']}</span>")
            if item.get("likes") is not None:
                meta.append(f"<span class=stat>❤️ {item['likes']}</span>")
            if item.get("updatedAt"):
                meta.append(f"<span class=meta>Updated: {item['updatedAt']}</span>")
            desc = (item.get("description") or "").strip()
            if len(desc) > 200:
                desc = desc[:197] + "..."
            return (
                "<div class=card>"
                f"<div class=card-title><a href='{item.get('url')}' target=_blank rel=noopener>{item.get('id')}</a> {badge}</div>"
                f"<div class=card-subtitle>{item.get('type','')} • {item.get('owner','')}</div>"
                f"<div class=card-desc>{desc}</div>"
                f"<div class=card-meta>{' | '.join(meta)}</div>"
                "</div>"
            )

        def section(title_text: str, items: List[Dict]) -> str:
            if not items:
                return ""
            cards = "\n".join(card_html(it) for it in items)
            return f"<section><h2>{title_text}</h2><div class=cards>{cards}</div></section>"

        html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Hugging Search — Dashboard</title>
  <style>
    :root {{ --bg:#0b0d12; --fg:#e6e9ef; --muted:#9aa4b2; --card:#121621; --accent:#5ac8fa; --warn:#eab308; }}
    body {{ background:var(--bg); color:var(--fg); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Inter, Arial, sans-serif; margin:0; padding:24px; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    .header {{ display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom: 16px; }}
    .title {{ font-size: 22px; margin: 0; }}
    .subtitle {{ color: var(--muted); }}
    .stats {{ display:flex; gap:10px; flex-wrap:wrap; margin: 8px 0 18px; }}
    .stat-chip {{ background: var(--card); border: 1px solid rgba(255,255,255,0.08); border-radius: 999px; padding: 6px 10px; font-size: 12px; color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px,1fr)); gap: 12px; }}
    .card {{ background: var(--card); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 12px; }}
    .card-title {{ font-weight: 600; margin-bottom: 4px; overflow-wrap:anywhere; }}
    .card-subtitle {{ color: var(--muted); font-size: 12px; margin-bottom: 8px; }}
    .card-desc {{ font-size: 13px; line-height: 1.45; min-height: 28px; margin-bottom: 8px; color: #d2d7df; }}
    .card-meta {{ font-size: 12px; color: var(--muted); display:flex; flex-wrap:wrap; gap:8px; }}
    .badge {{ background: rgba(234, 179, 8, 0.15); color: #facc15; border:1px solid rgba(250,204,21,0.35); border-radius: 999px; padding: 2px 8px; font-size: 11px; margin-left: 6px; }}
    h2 {{ font-size: 16px; margin: 18px 0 8px; color: var(--accent); }}
    .actions {{ display:flex; gap:8px; align-items:center; }}
    button {{ background:#1f2937;color:#e5e7eb;border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:8px 10px;cursor:pointer; }}
  </style>
  <script>
    function printToPDF() {{ window.print(); }}
  </script>
</head>
<body>
  <div class=\"container\">
    <div class=\"header\">
      <div>
        <div class=\"title\">Hugging Search — Dashboard</div>
        <div class=\"subtitle\">Query: {query}</div>
      </div>
      <div class=\"actions\"><button onclick=\"printToPDF()\">Print to PDF</button></div>
    </div>
    <div class=\"stats\">
      <div class=\"stat-chip\">Models: {len(models)}</div>
      <div class=\"stat-chip\">Datasets: {len(datasets)}</div>
      <div class=\"stat-chip\">Spaces: {len(spaces)}</div>
      <div class=\"stat-chip\">Daily papers: {len(papers) if isinstance(papers,list) else 0}</div>
    </div>
    {section("Models", models)}
    {section("Datasets", datasets)}
    {section("Spaces", spaces)}
  </div>
</body>
</html>
"""
        return html


