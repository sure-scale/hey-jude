# Contributing

Hey Jude is a privacy gateway, so correctness and data handling matter.

- Open an issue first for anything non-trivial.
- Security / privacy-leak findings go through [SECURITY.md](SECURITY.md), not a public issue.
- Read [AGENTS.md](AGENTS.md) — its laws are binding.

## The core rule (`AGENTS.md`)

No edge-case handling: no salvage parsing, retry loops, fallback branches,
defensive `try/except`, or graceful degradation. When something fails, **fail
loud** and fix the root cause. Tests assert exact behavior, nothing else.

## Setup

```bash
cp .env.example .env
uv sync --extra dev
uv run pytest          # full suite
uv run pytest -m e2e   # starts local services / calls providers
```

## Data handling

- **Never commit real PII, secrets, or keys** — code, tests, fixtures, or commits. Use synthetic data.
- `.env` is git-ignored. Document new keys in `.env.example` with placeholders.

## Pull requests

- One logical change per PR.
- Branches: `type/short-description` (e.g. `fix/mapping-collision`).
- Commits: [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `perf:`, `chore:`, `docs:`).
- Fill out the PR template; keep tests green.

Contributions are licensed under [AGPL-3.0](LICENSE).
