# Security & privacy

## Principles
- Read‑only: tools never send Authorization headers
- Respect gated/private resources and label them as not accessible
- Don’t log secrets; `HF_TOKEN` is only for the inference model

## Details
- Tools normalize `visibility` and `access` fields
- The Report view renders HTML in memory; no report files are saved


## Scope
- No write operations to the Hub
- Only public endpoints and domain‑restricted search are used
