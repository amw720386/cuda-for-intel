free-run token sequences: auto 96 tokens, none 96 tokens; first differing generated token index: 38

### Teacher-forced on the none-generated sequence (96 steps, reference dtype float32)

steps: 96; top-1 flips auto vs none: 1; auto top-1 != reference top-1: 2; none top-1 != reference top-1: 1

| quantity (per step, then over the steps) | max over steps | mean over steps |
|---|---|---|
| auto vs fp32 ref, max over vocab | 1.06579 | 0.54478 |
| none vs fp32 ref, max over vocab | 0.98566 | 0.54858 |
| auto vs ref, ref top-20 tokens | 0.53132 | 0.23950 |
| none vs ref, ref top-20 tokens | 0.51422 | 0.23993 |
| auto vs ref, RMS over vocab | 0.23347 | 0.11233 |
| none vs ref, RMS over vocab | 0.19045 | 0.11154 |
| auto vs none, RMS over vocab | 0.18151 | 0.09497 |
| auto vs none, max over vocab | 0.87977 | 0.45394 |
| auto vs none, ref top-20 tokens | 0.51938 | 0.21087 |

| step | token fed next | ref top-2 margin | auto margin | none margin | max abs dlogit auto-none (step) | auto top1 | none top1 | ref top1 |
|---|---|---|---|---|---|---|---|---|
| 38 | 22096 | 0.03018 | 0.07934 | 0.00278 | 0.31831 | 61572 | 22096 | 22096 |

Eight smallest reference top-2 margins of the sequence:

| step | ref margin | auto==none top1 | max abs dlogit auto-none |
|---|---|---|---|
| 38 | 0.03018 | NO | 0.31831 |
| 61 | 0.04894 | yes | 0.40223 |
| 30 | 0.06500 | yes | 0.47770 |
| 88 | 0.12254 | yes | 0.39592 |
| 64 | 0.13065 | yes | 0.36364 |
| 47 | 0.13613 | yes | 0.52126 |
| 63 | 0.18303 | yes | 0.41598 |
| 73 | 0.18936 | yes | 0.42269 |

Check: largest auto-vs-none logit difference at any step = 0.87977. Flips with reference margin > 2x that = 0.

Variants on the none-generated sequence (each row is a separate run, everything else equal):

| variant (CUDA_FOR_INTEL_SUBST) | steps | top-1 flips vs none | top-1 != ref top-1 | max abs dlogit vs fp32 ref (vocab) | same, ref top-20 | max abs dlogit vs none (vocab) | first step differing from none | max / mean(step max) abs dlogit vs CPU q8 pipeline | RMS vs ref | RMS vs none |
|---|---|---|---|---|---|---|---|---|---|---|
| auto | 96 | 1 | 2 | 1.06579 | 0.53132 | 0.87977 | 0 | 1.18432 / 0.48406 | 0.11233 | 0.09497 |
| none | 96 | 0 | 1 | 0.98566 | 0.51422 | 0.00000 | none (bit-identical) | 1.45536 / 0.49918 | 0.11154 | 0.00000 |
| cpu | 96 | 3 | 2 | 0.97834 | 0.57891 | 1.45536 | 0 | 0.00000 / 0.00000 | 0.11367 | 0.10532 |
| auto_nommq | 96 | 1 | 2 | 0.90540 | 0.50329 | 0.90384 | 9 | 1.45536 / 0.49261 | 0.11301 | 0.08032 |
| only_mmq | 96 | 0 | 1 | 1.11146 | 0.50396 | 0.93954 | 0 | 1.24936 / 0.49865 | 0.11656 | 0.09699 |
| only_mmvq | 96 | 0 | 1 | 0.98566 | 0.51422 | 0.00000 | none (bit-identical) | 1.45536 / 0.49918 | 0.11154 | 0.00000 |
| only_quant | 96 | 1 | 2 | 0.90540 | 0.50329 | 0.90384 | 9 | 1.45536 / 0.49261 | 0.11301 | 0.08032 |

### Teacher-forced on the auto-generated sequence (96 steps, reference dtype float32)

steps: 96; top-1 flips auto vs none: 1; auto top-1 != reference top-1: 2; none top-1 != reference top-1: 1

| quantity (per step, then over the steps) | max over steps | mean over steps |
|---|---|---|
| auto vs fp32 ref, max over vocab | 1.06579 | 0.53973 |
| none vs fp32 ref, max over vocab | 0.98566 | 0.54269 |
| auto vs ref, ref top-20 tokens | 0.53132 | 0.23501 |
| none vs ref, ref top-20 tokens | 0.46692 | 0.23058 |
| auto vs ref, RMS over vocab | 0.23347 | 0.11038 |
| none vs ref, RMS over vocab | 0.19045 | 0.11187 |
| auto vs none, RMS over vocab | 0.18151 | 0.09265 |
| auto vs none, max over vocab | 0.71321 | 0.44232 |
| auto vs none, ref top-20 tokens | 0.44171 | 0.20956 |

| step | token fed next | ref top-2 margin | auto margin | none margin | max abs dlogit auto-none (step) | auto top1 | none top1 | ref top1 |
|---|---|---|---|---|---|---|---|---|
| 38 | 61572 | 0.03018 | 0.07934 | 0.00278 | 0.31831 | 61572 | 22096 | 22096 |

Eight smallest reference top-2 margins of the sequence:

| step | ref margin | auto==none top1 | max abs dlogit auto-none |
|---|---|---|---|
| 38 | 0.03018 | NO | 0.31831 |
| 61 | 0.04429 | yes | 0.45247 |
| 30 | 0.06500 | yes | 0.47770 |
| 85 | 0.09564 | yes | 0.33582 |
| 87 | 0.19014 | yes | 0.45706 |
| 6 | 0.19114 | yes | 0.51273 |
| 7 | 0.23107 | yes | 0.40580 |
| 47 | 0.26178 | yes | 0.41639 |

Check: largest auto-vs-none logit difference at any step = 0.71321. Flips with reference margin > 2x that = 0.

Variants on the auto-generated sequence (each row is a separate run, everything else equal):

| variant (CUDA_FOR_INTEL_SUBST) | steps | top-1 flips vs none | top-1 != ref top-1 | max abs dlogit vs fp32 ref (vocab) | same, ref top-20 | max abs dlogit vs none (vocab) | first step differing from none | max / mean(step max) abs dlogit vs CPU q8 pipeline | RMS vs ref | RMS vs none |
|---|---|---|---|---|---|---|---|---|---|---|
| auto | 96 | 1 | 2 | 1.06579 | 0.53132 | 0.71321 | 0 | 1.18432 / 0.47506 | 0.11038 | 0.09265 |
| none | 96 | 0 | 1 | 0.98566 | 0.46692 | 0.00000 | none (bit-identical) | 1.45536 / 0.48784 | 0.11187 | 0.00000 |
| cpu | 96 | 2 | 1 | 0.97834 | 0.57891 | 1.45536 | 0 | 0.00000 / 0.00000 | 0.11138 | 0.10372 |
| auto_nommq | 96 | 1 | 2 | 0.85506 | 0.38671 | 0.64748 | 9 | 1.45536 / 0.48344 | 0.10920 | 0.07821 |
| only_mmq | 96 | 0 | 1 | 1.11146 | 0.52142 | 0.77256 | 0 | 1.24936 / 0.49692 | 0.11250 | 0.09455 |
| only_mmvq | 96 | 0 | 1 | 0.98566 | 0.46692 | 0.00000 | none (bit-identical) | 1.45536 / 0.48784 | 0.11187 | 0.00000 |
| only_quant | 96 | 1 | 2 | 0.85506 | 0.38671 | 0.64748 | 9 | 1.45536 / 0.48344 | 0.10920 | 0.07821 |
