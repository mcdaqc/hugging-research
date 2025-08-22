# Hugging Research — Documentation

Hugging Research is a lightweight research and coding assistant focused on Hugging Face Hub content. It helps you find models, datasets, Spaces, users, collections, and papers, and organizes links into a clean report.

- UI: Gradio Blocks
- Agent: smolagents `CodeAgent`
- Tools: `scripts/hf_tools.py` (anonymous, read‑only)
- Report: server-side generated from the final answer (no model call required)

## Contents
- [Getting started](./getting-started.md)
- [Architecture](./architecture.md)
- [Tools](./tools.md)
- [Security](./security.md)
- [Troubleshooting](./troubleshooting.md)
- [Contributing](./contributing.md)
- [Roadmap](./roadmap.md)

## What it does
- Searches Hugging Face Hub (models/datasets/Spaces/papers/users/collections)
- Pulls tutorials/blog/course content via domain‑restricted search when needed
- Avoids hallucinated links: only cites URLs from tool outputs
- Builds an HTML report of links in the Report view automatically

## How it works (quick view)
- The agent uses `hf_*` tools (return JSON as strings) and `web_search` (returns plain text)
- The app converts the final answer into a categorized link report
- No files are written to disk for reports; HTML is rendered in‑app

Start here: [Getting started](./getting-started.md)


