from typing import List, Optional, Dict
from smolagents import Tool

class HFLinkReportTool(Tool):
    """Generate a single-layout HTML report (cards + counters) from a final textual answer.
    The tool extracts links from the provided text, categorizes them (HF models/datasets/spaces/papers, blogs, repos, videos, news),
    and renders a consistent link report. Always returns a full HTML document (starts with <!DOCTYPE html>)."""

    name = "hf_links_to_report"
    description = (
        "Create an HTML report from a final answer text. The tool parses links, groups them into categories "
        "(Hugging Face models/datasets/spaces/papers and external resources like blogs/repos/videos/news), and renders cards. "
        "Inputs: final_answer (string, required), query (string, optional), title (string, optional). Returns an HTML document."
    )
    inputs = {
        "final_answer": {"type": "string", "description": "Final answer text containing inline links"},
        "query": {"type": "string", "description": "Original user intent or topic", "nullable": True},
        "title": {"type": "string", "description": "Dashboard title", "nullable": True},
    }
    output_type = "string"

    def forward(self, final_answer: str, query: Optional[str] = None, title: Optional[str] = None) -> str:
        try:
            import re
            import json as _json
            doc_title = title or "Report"
            query = (query or "").strip()
            header_html = f"<div class=\"header\"><div><div class=\"title\">{title}</div></div></div>" if title else ""

            # Extract URLs
            urls = re.findall(r"https?://[^\s)\]]+", final_answer or "")
            # Categorize
            cats = {
                "models": [], "datasets": [], "spaces": [], "papers": [],
                "blogs": [], "repos": [], "videos": [], "news": [], "other": []
            }
            for u in urls:
                low = u.lower()
                if "huggingface.co/" in low:
                    # Prefer explicit kinds first to avoid misclassifying /datasets/* as generic owner/repo
                    if "/datasets/" in low:
                        cats["datasets"].append(u)
                    elif "/spaces/" in low:
                        cats["spaces"].append(u)
                    elif "/papers/" in low:
                        cats["papers"].append(u)
                    elif "/models/" in low:
                        cats["models"].append(u)
                    else:
                        # Treat bare owner/repo as models only if it is NOT under known sections
                        # e.g., huggingface.co/owner/repo → model repo; huggingface.co/blog/... → blog
                        m = re.search(r"huggingface\.co/([^/]+)/([^/]+)$", low)
                        if m and m.group(1) not in {"datasets", "spaces", "papers", "blog", "learn", "docs", "organizations", "collections"}:
                            cats["models"].append(u)
                        else:
                            cats["blogs"].append(u)
                elif "github.com" in low:
                    cats["repos"].append(u)
                elif "youtube.com" in low or "youtu.be" in low:
                    cats["videos"].append(u)
                elif any(d in low for d in ["arxiv.org", "medium.com", "towardsdatascience.com", "huggingface.co/blog", "huggingface.co/learn"]):
                    cats["blogs"].append(u)
                elif any(d in low for d in ["theverge.com", "techcrunch.com", "venturebeat.com", "wired.com", "mit.edu"]):
                    cats["news"].append(u)
                else:
                    cats["other"].append(u)

            def chips_section():
                chips = [
                    ("Models", len(cats["models"])),
                    ("Datasets", len(cats["datasets"])),
                    ("Spaces", len(cats["spaces"])),
                    ("Papers", len(cats["papers"])),
                    ("Blogs/Docs", len(cats["blogs"])),
                    ("Repos", len(cats["repos"])),
                    ("Videos", len(cats["videos"])),
                    ("News", len(cats["news"]))
                ]
                return "\n".join([f"<div class=stat-chip>{name}: {count}</div>" for name, count in chips])

            def host_icon(host: str) -> str:
                return ""

            def card_list(urls: List[str], data_cat: str) -> str:
                items = []
                for u in urls:
                    host = re.sub(r"^https?://", "", u).split("/")[0]
                    icon = host_icon(host)
                    favicon = f"https://www.google.com/s2/favicons?sz=32&domain={host}"
                    items.append(
                        f"<div class=card data-cat='{data_cat}'>"
                        f"<div class=card-title>{icon} <img class=\"fav\" src=\"{favicon}\" alt=\"\"/> <a href='{u}' target=_blank rel=noopener>{u}</a></div>"
                        f"<div class=card-subtitle>{host}</div>"
                        f"<div class=card-actions><button onclick=\"copyLink('{u}')\">Copy</button></div>"
                        "</div>"
                    )
                return "\n".join(items)

            def section(title_text: str, urls: List[str], key: str) -> str:
                if not urls:
                    return ""
                return f"<section data-key='{key}'><h2>{title_text}</h2><div class=cards>{card_list(urls, key)}</div></section>"

            html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{doc_title}</title>
  <style>
    :root {{ --bg:#0b0d12; --fg:#e6e9ef; --muted:#9aa4b2; --card:#121621; --accent:#5ac8fa; }}
    body {{ background:var(--bg); color:var(--fg); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Inter, Arial, sans-serif; margin:0; padding:24px; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    .header {{ display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom: 12px; }}
    .title {{ font-size: 22px; margin: 0; }}
    .subtitle {{ color: var(--muted); }}
    .stats {{ display:flex; gap:10px; flex-wrap:wrap; margin: 8px 0 18px; }}
    .stat-chip {{ background: var(--card); border: 1px solid rgba(255,255,255,0.08); border-radius: 999px; padding: 6px 10px; font-size: 12px; color: var(--muted); }}
    h2 {{ font-size: 16px; margin: 18px 0 8px; color: var(--accent); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px,1fr)); gap: 12px; }}
    .card {{ background: var(--card); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 12px; }}
    .card-title {{ font-weight: 600; margin-bottom: 4px; overflow-wrap:anywhere; }}
    .card-subtitle {{ color: var(--muted); font-size: 12px; }}
    .answer {{ line-height:1.55; color:#d2d7df; }}
    .card-actions button {{ background:#1f2937;color:#e5e7eb;border:1px solid rgba(255,255,255,0.08);border-radius:6px;padding:4px 8px;cursor:pointer;font-size:12px; }}
    .fav {{ width:14px; height:14px; vertical-align:middle; margin-right:6px; border-radius:4px; }}
    .warn {{ margin-left:6px; cursor: help; }}
  </style>
  <script src=\"https://cdn.jsdelivr.net/npm/marked/marked.min.js\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js\"></script>
</head>
<body>
  <div class=\"container\">{header_html}
    <h2>You may be interested <span class=\"warn\" title=\"Links may be AI‑generated and might not resolve.\">⚠️</span></h2>
    <div class=\"stats\">{chips_section()}</div>
    {section('Models', cats['models'], 'models')}
    {section('Datasets', cats['datasets'], 'datasets')}
    {section('Spaces', cats['spaces'], 'spaces')}
    {section('Papers', cats['papers'], 'papers')}
    {section('Blogs / Docs', cats['blogs'], 'blogs')}
    {section('Repositories', cats['repos'], 'repos')}
    {section('Videos', cats['videos'], 'videos')}
    {section('News', cats['news'], 'news')}
    {section('Other', cats['other'], 'other')}
  </div>
  <script>
    function copyLink(url){{ try{{navigator.clipboard && navigator.clipboard.writeText(url);}}catch(e){{}} }}
  </script>
</body>
</html>
"""
            return html
        except Exception as e:
            return f"<!DOCTYPE html><html><body><pre>Error generating report: {str(e)}</pre></body></html>"