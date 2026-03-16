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

### README
- **Update README.md whenever a feature is added or changed** — keep the features list, supported backends table, project structure, and usage examples in sync with the code
- Do this in the same PR as the feature, not as a follow-up

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

<!-- mulch:start -->
## Project Expertise (Mulch)
<!-- mulch-onboard-v:1 -->

This project uses [Mulch](https://github.com/jayminwest/mulch) for structured expertise management.

**At the start of every session**, run:
```bash
mulch prime
```

This injects project-specific conventions, patterns, decisions, and other learnings into your context.
Use `mulch prime --files src/foo.ts` to load only records relevant to specific files.

**Before completing your task**, review your work for insights worth preserving — conventions discovered,
patterns applied, failures encountered, or decisions made — and record them:
```bash
mulch record <domain> --type <convention|pattern|failure|decision|reference|guide> --description "..."
```

Link evidence when available: `--evidence-commit <sha>`, `--evidence-bead <id>`

Run `mulch status` to check domain health and entry counts.
Run `mulch --help` for full usage.
Mulch write commands use file locking and atomic writes — multiple agents can safely record to the same domain concurrently.

### Before You Finish

1. Discover what to record:
   ```bash
   mulch learn
   ```
2. Store insights from this work session:
   ```bash
   mulch record <domain> --type <convention|pattern|failure|decision|reference|guide> --description "..."
   ```
3. Validate and commit:
   ```bash
   mulch sync
   ```
<!-- mulch:end -->

<!-- seeds:start -->
## Issue Tracking (Seeds)
<!-- seeds-onboard-v:1 -->

This project uses [Seeds](https://github.com/jayminwest/seeds) for git-native issue tracking.

**At the start of every session**, run:
```
sd prime
```

This injects session context: rules, command reference, and workflows.

**Quick reference:**
- `sd ready` — Find unblocked work
- `sd create --title "..." --type task --priority 2` — Create issue
- `sd update <id> --status in_progress` — Claim work
- `sd close <id>` — Complete work
- `sd dep add <id> <depends-on>` — Add dependency between issues
- `sd sync` — Sync with git (run before pushing)

### Before You Finish
1. Close completed issues: `sd close <id>`
2. File issues for remaining work: `sd create --title "..."`
3. Sync and push: `sd sync && git push`
<!-- seeds:end -->

<!-- canopy:start -->
## Prompt Management (Canopy)
<!-- canopy-onboard-v:1 -->

This project uses [Canopy](https://github.com/jayminwest/canopy) for git-native prompt management.

**At the start of every session**, run:
```
cn prime
```

This injects prompt workflow context: commands, conventions, and common workflows.

**Quick reference:**
- `cn list` — List all prompts
- `cn render <name>` — View rendered prompt (resolves inheritance)
- `cn emit --all` — Render prompts to files
- `cn update <name>` — Update a prompt (creates new version)
- `cn sync` — Stage and commit .canopy/ changes

**Do not manually edit emitted files.** Use `cn update` to modify prompts, then `cn emit` to regenerate.
<!-- canopy:end -->
