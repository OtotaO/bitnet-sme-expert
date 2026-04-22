# Semantic Versioning Policy

This repository follows **Semantic Versioning 2.0.0** to provide predictable upgrades for downstream users.

## Version format

`MAJOR.MINOR.PATCH` (example: `2.3.4`)

## Increment rules

- **MAJOR**: Increment for backward-incompatible API or behavior changes.
- **MINOR**: Increment for backward-compatible feature additions.
- **PATCH**: Increment for backward-compatible bug fixes and internal hardening.

## Stability and compatibility expectations

- Public API contract changes require clear migration notes in release notes.
- Deprecations should be announced in one minor release before removal when practical.
- Critical fixes may be released as patch versions and should not break existing integrations.

## Release process expectations

1. Update `CHANGELOG.md` from `Unreleased` entries.
2. Tag release using `vMAJOR.MINOR.PATCH`.
3. Publish release notes that include:
   - user-visible changes,
   - migration/rollback guidance,
   - links to CI/security/load-test artifacts.
4. Ensure `README.md` references the current release note URL.

## Pre-release tags

Pre-release identifiers (e.g., `-rc.1`, `-beta.2`) may be used for validation builds.
These are not considered stable for production unless explicitly stated.
