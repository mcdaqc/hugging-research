# Architecture

## Overview
Gradio UI with a single Report view. A smolagents `CodeAgent` uses `hf_*` tools to fetch public Hub data, then the app turns the final answer into a categorized HTML link report.

## Components
- `app.py`
  - Session and API key setup (for inference model only)
  - Streams agent messages
  - Generates report HTML from the final answer (no file writes)
- `scripts/hf_tools.py`
  - Anonymous, read‑only wrappers of Hub APIs and domain‑restricted search
  - Outputs JSON as strings
- `scripts/report_generator.py`
  - Parses links in the final answer and renders a self‑contained HTML report

## Flow
1. User sends a prompt
2. Agent calls `hf_*` tools and composes an answer with inline links
3. App converts that answer into an HTML link report and shows it in the Report view

## Privacy
- Tools never use tokens; gated/private items are marked as not accessible
- `HF_TOKEN` is only for the inference model

## Extending
- Add tools in `scripts/hf_tools.py` and register in `create_tools_with_model`
- Update the system prompt to document tool contracts
