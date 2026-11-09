# X51 controlled scaling and memory sweep (EXPLORATORY, descriptive)

93 synthetic scenarios (data/make_scale_corpus.py), profiled with the harness pipeline; one scenario per task; every task solo (`--solo 4`), JVM `-Xmx16g -XX:+UseSerialGC`, no -Xms. Peak RSS is the JVM process's resident peak (wait4): the 2 GiB ergonomic initial heap is reserved but not all touched, so RSS here is real use plus GC slack (an upper bound on SF's live memory), for ONE scenario per task. Initial matrix: levenshtein (stage 1). PCG = propagation graph; times from SFSTATS (build, iterate) unless marked as task wall time.

**Design limit, visible in the FD table:** planted derived columns are visible as FDs only where their parents are on the same side, so the measured FD count per side grows with that side's width but falls with the other side's (the universe widens). Counts stay at most about 80 per side; Valentine's tables have far more. The FD arms therefore add few PCG elements here, and the sweep measures the encoding's scaling at low FD counts.

## Failures / OOM

- melnik: 0 of 93 without output
- fd: 0 of 93 without output
- fd_reified: 0 of 93 without output

## Measured FD counts (grid 1, median over replicates)

| n \ m | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| 16 | 2 / 3 | 1 / 7 | 2 / 17 | 0 / 40 | 0 / 79 |
| 32 | 7 / 2 | 5 / 5 | 3 / 13 | 1 / 38 | 0 / 72 |
| 64 | 18 / 0 | 15 / 5 | 11 / 11 | 6 / 29 | 2 / 64 |
| 128 | 36 / 0 | 35 / 3 | 25 / 3 | 19 / 23 | 11 / 59 |
| 256 | 77 / 1 | 73 / 1 | 64 / 3 | 55 / 8 | 41 / 40 |

FD_s against n: k = 1.53 (R² 0.83); counts are measured (planted FDs are visible only where a derived column's parents are on the same side).

## P1-P4: graph size and time against n·m (grid 1)

| arm | PCG vertices k (R²) | PCG edges k (R²) | build ms vs PCG edges k (R²) | ms per iteration vs PCG edges k (R²) | median iterations | PCG edges at 16×16 / 256×256 | build ms at 16×16 / 256×256 |
|---|---|---|---|---|---|---|---|
| melnik | 0.93 (1.00) | 0.96 (1.00) | 0.44 (0.94) | 0.58 (0.96) | 19 | 2626 / 534628 | 723 / 9427 |
| fd | 0.93 (1.00) | 0.96 (1.00) | 0.44 (0.94) | 0.61 (0.93) | 19 | 2644 / 541286 | 668 / 8834 |
| fd_reified | 0.93 (1.00) | 0.96 (1.00) | 0.44 (0.94) | 0.64 (0.92) | 19 | 2868 / 592460 | 654 / 8619 |

**P2, FD-PCG edges** (fd − melnik PCG edges): against n·m k = 0.96 (R² 0.87, 56 scenarios with > 0); against |FD_s|·|FD_t| k = 1.03 (R² 0.98). Scenarios with 0 extra edges: 19.

## P5: unequal sizes at fixed n·m (melnik PCG edges, median over replicates)

| n·m | cells (n×m: edges) | max/min |
|---|---|---|
| 512 | 16×32: 5074, 32×16: 4930 | 1.029 |
| 1024 | 16×64: 9894, 32×32: 9426, 64×16: 9616 | 1.050 |
| 2048 | 16×128: 18834, 32×64: 18130, 64×32: 18406, 128×16: 19366 | 1.068 |
| 4096 | 16×256: 37266, 32×128: 35922, 64×64: 35428, 128×32: 36070, 256×16: 37266 | 1.052 |
| 8192 | 32×256: 70354, 64×128: 69476, 128×64: 69476, 256×32: 71398 | 1.028 |
| 16384 | 64×256: 137574, 128×128: 136294, 256×64: 137574 | 1.009 |
| 32768 | 128×256: 269926, 256×128: 269926 | 1.000 |

## FD density at 64×64 (grid 2, median over replicates)

| ρ | FD_s / FD_t | melnik PCG edges | fd PCG edges | fd − melnik | fd_reified PCG vertices | fd build ms | fd_reified build ms |
|---|---|---|---|---|---|---|---|
| 0.00 | 0 / 0 | 34882 | 34882 | 0 | 9129 | 2146 | 2163 |
| 0.15 | 8 / 5 | 35428 | 35588 | 160 | 10420 | 2083 | 2158 |
| 0.30 | 11 / 11 | 35428 | 35762 | 576 | 11138 | 2176 | 2099 |
| 0.45 | 15 / 15 | 35428 | 36438 | 1008 | 11956 | 2222 | 2271 |
| 0.60 | 19 / 28 | 35430 | 37066 | 1638 | 12796 | 2478 | 2239 |

## P6: rows at 64×64, ρ = 0.3 (grid 3, median over replicates)

| rows | FD_s / FD_t | melnik PCG edges | fd PCG edges | melnik build ms | fd build ms | datatype task s |
|---|---|---|---|---|---|---|
| 500 | 11 / 11 | 35428 | 35848 | 822 | 798 | 1.7 |
| 2000 | 11 / 11 | 35428 | 35762 | 2022 | 2176 | 3.2 |
| 8000 | 12 / 11 | 35154 | 35594 | 5818 | 5847 | 7.1 |

## P7: the additive term's cost against SF (grid 1, median over replicates)

The datatype matcher's cost is only available as its whole stage-1 task (JVM start, reading both tables, matching, writing): an upper bound. 'above 16×16' subtracts the 16×16 cell's median task time as a rough fixed-cost estimate. The arithmetic q + 5D is pure computation (TimeAdditive.java, median of 7).

| n×m | datatype task s | datatype above 16×16 s | q+5D ms | melnik build ms | melnik build+iterate ms | SF(melnik) build ÷ datatype task | arithmetic ÷ datatype task |
|---|---|---|---|---|---|---|---|
| 16×16 | 1.9 | 0.0 | 0.015 | 723 | 730 | 0.38 | 0.00001 |
| 32×32 | 2.5 | 0.6 | 0.010 | 1230 | 1250 | 0.49 | 0.00000 |
| 64×64 | 3.2 | 1.3 | 0.014 | 2022 | 2099 | 0.63 | 0.00000 |
| 128×128 | 4.6 | 2.7 | 0.053 | 3727 | 3941 | 0.81 | 0.00001 |
| 256×256 | 7.1 | 5.2 | 0.206 | 9427 | 10627 | 1.33 | 0.00003 |
| 16×256 | 5.0 | 3.1 | 0.051 | 3440 | 3517 | 0.69 | 0.00001 |
| 256×16 | 4.8 | 2.9 | 0.015 | 3550 | 3638 | 0.74 | 0.00000 |

Datatype task time against (n + m)·r over all 93 scenarios: k = 0.49 (R² 0.98); it includes the fixed JVM cost, which flattens the slope.

## P8: peak RSS (task, MB; median over replicates)

| n×m | melnik | fd | fd_reified | fd_reified PCG edges |
|---|---|---|---|---|
| 16×16 | 437 | 444 | 447 | 2868 |
| 32×32 | 667 | 667 | 673 | 10266 |
| 64×64 | 727 | 728 | 733 | 39832 |
| 128×128 | 783 | 790 | 787 | 151212 |
| 256×256 | 936 | 922 | 928 | 592460 |
| 16×256 | 771 | 771 | 775 | 40546 |
| 256×16 | 766 | 768 | 783 | 41408 |

Largest peak RSS of any task: 935 MB (fd_reified, SCn256m256d30r2000s1). melnik / fd maxima: 940 / 941 MB.

