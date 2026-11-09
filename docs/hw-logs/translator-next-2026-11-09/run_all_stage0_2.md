# hw-validate results

revision: 31f1c98 wr-int  overall: **review**  generated: 2026-11-09T06:28:30

| stage | title | status | secs |
|---|---|---|---|
| 0 | environment + build | pass | 21 |
| 2 | FA Ollama text check (CPU vs shim FA off vs FA on) | review | 1549 |

## stage 0: environment + build - pass
- [pass] disk-free-gb: 1581.1 GB free (need 30)
- [info] stale-processes: present (not touched, may belong to others): ollama:14164 ollama app:6284
- [info] gpu-queue: lock free
- [pass] ps1-parse-check: all scripts parse
- [pass] cargo-build-release: rc=0
- [pass] cargo-test-build: rc=0
- [info] dll-sha256: nvcuda.dll=E0CC9EB13940A5DE cublas64_12.dll=F353E51C13A75A6C
- [info] translator-version: source TRANSLATOR_VERSION=6 (expected 5 on hw-validate)

## stage 2: FA Ollama text check (CPU vs shim FA off vs FA on) - review
- [pass] fa-qwen-short: fa1==fa0:False fa1==cpu:True first_diff_vs_cpu=-1 text=' Paris is a city of contrasts, with a rich history and a vibrant culture that has captivat'
- [pass] fa-qwen-long: fa1==fa0:True fa1==cpu:True first_diff_vs_cpu=-1 text=' The capital of France is Paris. It is a city located in the south of France, known for it'
- [pass] fa-llama32-1b-q4km-short: fa1==fa0:False fa1==cpu:True first_diff_vs_cpu=-1 text=' Paris is a city of great historical significance, known for its stunning architecture, vi'
- [review] fa-llama32-1b-q4km-long: fa1==fa0:False fa1==cpu:False first_diff_vs_cpu=149 text=' The capital of France is Paris. Paris is a major city in France, known for its historic a'
  - blame: read the text: coherent diverging text is expected fp drift; gibberish = FA/translator regression
- [review] fa-qwen25-15b-q4km-short: fa1==fa0:True fa1==cpu:False first_diff_vs_cpu=109 text=' Paris is the capital and most populous city of France. It is located on the Seine River i'
  - blame: read the text: coherent diverging text is expected fp drift; gibberish = FA/translator regression
- [review] fa-qwen25-15b-q4km-long: fa1==fa0:True fa1==cpu:False first_diff_vs_cpu=96 text=' The capital of France is Paris. Paris is a city located on the Seine River in the northwe'
  - blame: read the text: coherent diverging text is expected fp drift; gibberish = FA/translator regression

