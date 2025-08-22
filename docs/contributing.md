# Contributing

## Branches and PRs
- Branch from `main` using `feature/...` or `fix/...` prefixes
- Keep PRs focused and small; include screenshots for UI changes
- Link related issues if applicable

## Code style
- Python 3.10+
- Descriptive names; functions as verbs, variables as nouns
- Handle errors explicitly; avoid silent excepts
- Keep indentation and formatting consistent with the repo

## Commit messages
- Prefix: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`
- Keep them concise and meaningful

## Tests / Manual checks
- Run the app and verify:
  - Desktop and mobile UIs load
  - Single Report view renders the link report from the last answer
  - HF tools return JSON strings and handle errors

## PR checklist
- [ ] Code builds and runs locally
- [ ] No secrets or tokens committed
- [ ] Updated docs if behavior/UI changed
- [ ] Included screenshots for UI updates
