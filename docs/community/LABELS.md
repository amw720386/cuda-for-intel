# Label taxonomy

Created by `scripts/community/create_labels.sh` (not yet run; it needs maintainer credentials). Colour/description are in that script.

| Group | Labels |
| --- | --- |
| area | `area:translator` (PTX to SPIR-V), `area:subst` (kernel substitution), `area:cublas`, `area:cudnn`, `area:ollama`, `area:pytorch`, `area:nvrtc`, `area:docs`, `area:build-ci` |
| hardware | `hw:arc-b390`, `hw:other-arc`, `hw:other-intel` (reports on other devices are welcome, tagged by device) |
| kind | `kind:bug`, `kind:feature`, `kind:question`, `kind:compat-report` |
| status | `status:needs-repro`, `status:needs-info`, `status:confirmed`, `status:wontfix`, `status:duplicate`, `status:stale` |
| help | `good-first-issue`, `help-wanted` |
| other | `breaking-change`, `security` (use private reporting, see SECURITY note), `release-blocker` |

Every issue should end up with one `kind:`, at least one `area:`, and a `status:`.
