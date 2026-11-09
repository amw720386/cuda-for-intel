Host: fake_cpu=False

| Level | Workload | Result | Seconds | Notes |
|---|---|---|---|---|
| L1 | tensor create/copy/elementwise | pass | 11.3 |  |
| L2 | matmul f32/f16 vs CPU | pass | 1.6 |  |
| L3 | softmax/layernorm/reductions | pass | 12.3 |  |
