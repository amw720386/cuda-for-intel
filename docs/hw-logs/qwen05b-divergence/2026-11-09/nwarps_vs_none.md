steps=96 vocab=151936; reference = SUBST=none free run, others teacher-forced on its tokens
| variant | SUBST | mean RMS dlogit vs none | max RMS | mean max-abs | top-1 flips vs none | none margin at flips |
|---|---|---|---|---|---|---|
| auto | [] | 0.07808 | 0.11419 | 0.3706 | 1 | 0.126@33 |
| only_mmq | [mmq] | 0.07808 | 0.11419 | 0.3706 | 1 | 0.126@33 |
| only_mmvq | [mmvq] | 0.00000 | 0.00000 | 0.0000 | 0 | - |
| only_quant | [quantize_q8_1] | 0.00000 | 0.00000 | 0.0000 | 0 | - |
| mmvq_nw1 | [mmvq] | 0.07535 | 0.12294 | 0.3609 | 1 | 0.013@19 |
| mmvq_nw2 | [mmvq] | 0.07717 | 0.12434 | 0.3670 | 2 | 0.114@25, 0.034@74 |
| mmvq_nw8 | [mmvq] | 0.07297 | 0.14030 | 0.3541 | 4 | 0.114@25, 0.126@33, 0.085@67, 0.034@74 |
