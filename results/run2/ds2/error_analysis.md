## DS2 error analysis (primary ensemble, raw argmax; from saved predictions)

**S is dominated by record 232:** 1,381 of 1,836 S beats (75%).

| S beats | n | Se | PPV |
|---|---|---|---|
| all DS2 | 1,836 | 0.186 | 0.560 |
| record 232 only | 1,381 | 0.093 | 1.000 |
| DS2 without 232 | 455 | 0.470 | 0.443 |

Record 232's S beats were predicted as: N 674, S 128, V 579, F 0.

**F is dominated by record 213:** 362 of 388 F beats; predicted as N 350, S 0, V 12, F 0.

**N beats called F: 745.** Largest sources: 214 (318), 117 (184), 202 (127), 228 (44).

**N beats called V: 572.** Largest sources: 111 (134), 233 (113), 105 (86), 214 (68).

**Secondary (fixed 12 epochs) vs primary (early-stopped), paired record bootstrap:** macro-F1 difference +0.060, 95% CI [-0.037, +0.103], secondary higher in 74% of resamples. S Se on record 232: 0.338 (primary 0.093); on the other records: 0.611 (primary 0.470). Reported for comparison only; the primary result was fixed before DS2 was opened.

