# CLAUDE.md

## GitHub Workflow

### Branching
- `main` is the default branch — all work merges here
- Feature branches use the pattern `feature/<short-description>` (e.g. `feature/llamacpp-backend`)
- Always create a branch, open a PR, and merge — never push directly to main

### Issues
- Every planned piece of work gets a GitHub issue before starting
- Issues are labeled for discoverability:
  - **Category:** `backend`, `cli`, `testing`, `performance`
  - **Type:** `feature`, `bug`, `enhancement`
  - **Priority:** `priority: high`, `priority: low`
- Issue bodies include a Description and Requirements section

### Pull Requests
- PR title: short, under 70 chars
- PR body format:
  - `## Summary` — bullet points of what changed
  - `## Test plan` — checklist of how it was verified
  - `Closes #N` to auto-close the issue on merge
- **Always wait for CI to pass before merging** — use `gh run watch` to block until the run completes, then verify success before calling `gh pr merge`
- Use merge commits (not squash or rebase)
- Delete branch after merge

### Commits
- Commit messages: imperative mood, first line summarizes the change
- Body explains *why*, not just *what*
- Always include `Co-Authored-By` trailer when AI-assisted
- One logical change per commit — fix lint separately from features

### CI
- GitHub Actions workflow at `.github/workflows/ci.yml`
- Runs on push to main and on PRs targeting main
- Matrix: Python 3.11 + 3.12
- Steps: install deps, ruff lint, pytest

## Project

- **Language:** Python 3.11+
- **Linter:** ruff (config in pyproject.toml, rules: E/F/W/I)
- **Tests:** pytest, test files in `tests/`
- **Build:** setuptools via pyproject.toml
- **Repo:** https://github.com/overtimepog/froggy
