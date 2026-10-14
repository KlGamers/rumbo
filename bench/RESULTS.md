# CLINC150 domain routing

- routes: 10 domains, 15000 examples
- thresholds tuned on: 3100 validation cases
- reported on: 5500 test cases, 1000 out of scope
- machine: arm64, Darwin, Python 3.12.15

| router | accuracy | in-scope | out-of-scope recall | macro F1 | p50 ms | p95 ms | escalated |
|---|---|---|---|---|---|---|---|
| centroid / hash | 0.6582 | 0.8044 | 0.0000 | 0.6654 | 0.07 | 0.11 | - |
| kNN / hash | 0.8253 | 0.9364 | 0.3250 | 0.8397 | 2.62 | 2.87 | - |
| kNN / hash + out-of-scope examples | 0.8255 | 0.9364 | 0.3260 | 0.8399 | 2.65 | 2.93 | - |
| cascade / hash: centroid then kNN | 0.8225 | 0.9302 | 0.3380 | 0.8369 | 2.63 | 2.85 | 62.0% |
| centroid / static | 0.6458 | 0.7860 | 0.0150 | 0.6498 | 0.07 | 0.09 | - |
| kNN / static | 0.8175 | 0.9473 | 0.2330 | 0.8307 | 0.46 | 0.51 | - |
| kNN / static + out-of-scope examples | 0.8180 | 0.9473 | 0.2360 | 0.8313 | 0.48 | 0.56 | - |
| cascade / static: centroid then kNN | 0.8155 | 0.9436 | 0.2390 | 0.8287 | 0.54 | 0.60 | 72.8% |
