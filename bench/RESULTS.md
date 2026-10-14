# CLINC150 domain routing

- routes: 10 domains, 15000 examples
- thresholds and l2 tuned on: 3100 validation cases
- reported on: 5500 test cases, 1000 out of scope
- machine: arm64, Darwin, Python 3.12.15
- linear / hash: l2=1e-05, cut=0.341, validation accuracy 0.9435
- linear / onnx + hash: l2=1e-05, cut=0.434, validation accuracy 0.9623

| router | accuracy | in-scope | out-of-scope recall | macro F1 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| centroid / hash | 0.6913 | 0.8449 | 0.0000 | 0.6966 | 0.09 | 0.12 |
| kNN / hash | 0.8116 | 0.9313 | 0.2730 | 0.8251 | 9.02 | 11.44 |
| kNN / static | 0.8180 | 0.9473 | 0.2360 | 0.8313 | 0.47 | 0.82 |
| linear / hash | 0.8715 | 0.9622 | 0.4630 | 0.8868 | 0.14 | 0.17 |
| linear / onnx + hash | 0.8925 | 0.9760 | 0.5170 | 0.9066 | 4.67 | 6.92 |
