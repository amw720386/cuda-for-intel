# Prefill with kernel substitution off vs on, 2026-11-09 (claims-audit row 6)

Arc B390, driver 32.0.101.9033, stock Ollama v0.12.6, flash attention off, LD16 off, 624-token prompt (623 for Llama) after a
different warm-up prompt of the same length (no KV prefix cache hit), single runs. Build: main + translator-next + win-redirect
(`TRANSLATOR_VERSION` 6, the integration build tested the same day). `scripts/profile/ollama_profile.ps1 -NoProfiled -Fa 0`, once
with `CUDA_FOR_INTEL_SUBST=none` and once with `auto`; raw: `timings_subst_none.json`, `timings_subst_auto.json`.

| model | SUBST none | SUBST auto | ratio | README chart (release candidate 7fc40fb) |
|---|---|---|---|---|
| Qwen2.5-0.5B Q8_0 | 1814 | 6876 | 3.8x | 1818 to 6767 (3.7x) |
| Llama-3.2-1B Q4_K_M | 924 | 5258 | 5.7x | 817 to 4877 (6.0x) |
| Qwen2.5-1.5B Q4_K_M | 635 | 4028 | 6.3x | 627 to 3727 (5.9x) |

Decode (64 tokens) none / auto: 36.7 / 42.3, 34.6 / 37.9, 23.9 / 28.7 tok/s. The chart's numbers are reproduced within about 10 %
(each pair is one run on a shared machine).
