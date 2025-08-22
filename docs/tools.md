# Tools

All `hf_*` tools are anonymous, read‑only, and return JSON as a string.
- Listing tools → `{ "results": [...] }`
- Detail tools → `{ "item": {...} }`

Common metadata: `visibility`, `access`, `type`, `id`, `owner`, `url`, `likes`, `downloads`, `updatedAt`.

## Searches
- `hf_models_search(query, owner?, task?, tags?, sort?, direction?, limit=10)`
- `hf_datasets_search(query, owner?, tags?, sort?, direction?, limit=10)`
- `hf_spaces_search(query, owner?, tags?, sort?, direction?, limit=10)`
- `hf_site_search(query, limit=10)` — Blog/Learn/Docs discovery (plain links)

## Details
- `hf_model_info(repo_id)`
- `hf_dataset_info(repo_id)`
- `hf_space_info(repo_id)`
- `hf_user_info(username)`
- `hf_collections_list(owner)` / `hf_collection_get(owner, slug)`
- `hf_paper_info(arxiv_id)` / `hf_paper_repos(arxiv_id)` / `hf_daily_papers(date?)`
- `hf_repo_info(repo_type, repo_id)`

## Parsing rules (important)
- `web_search` → returns plain text. Do not `json.loads`.
- `hf_*` tools → return JSON as a string. Always `json.loads(...)` before indexing.

## Examples
- “Find top Stable Diffusion models.”
- “Datasets for Spanish sentiment analysis.”
- “Spaces for document Q&A.”
