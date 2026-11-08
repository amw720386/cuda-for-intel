# Governance (DRAFT)

## Roles
- **Owner**: holds the repository, releases, licence and trademark decisions.
- **Maintainers**: merge PRs, triage, run the hardware suite and releases (RELEASING.md). Added by the owner after sustained, reviewed contributions and a request. Inactive for 12 months: moved to emeritus.
- **Contributors**: anyone with a merged or proposed change, issue or compatibility report.

## Decisions
- Day to day: lazy consensus on PRs; one maintainer approval to merge, two for changes to the public contract in VERSIONING.md (env switches, DLL ABI, cache format) or to CI/release files.
- Disagreement: discussion in the issue/PR; the owner decides after 7 days.
- Hardware claims need a log in `docs/hw-logs/` (existing project rule). Nothing is advertised as working without one.
- Licence, trademark, and CLA/DCO changes: owner only.

## Code of conduct
Adopt the Contributor Covenant v2.1 (https://www.contributor-covenant.org/version/2/1/code_of_conduct/) as `CODE_OF_CONDUCT.md` (not added here: it needs a real enforcement contact, which the owner must supply as a private address or GitHub private reporting; do not publish a personal address by accident). Adoption note to paste into README/CONTRIBUTING: "Participation is governed by our Code of Conduct (Contributor Covenant 2.1)."

## Security
Vulnerabilities go through GitHub private vulnerability reporting (the SECURITY note belongs to the packaging branch). Never in public issues.

## Changes to this file
By PR, owner approval.
