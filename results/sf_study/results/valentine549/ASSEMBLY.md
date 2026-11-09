# Valentine 549 assembly (2026-11-03T00:20:56, reference/assemble_549.py)

Base 549 = 381 old (the Joinable copy of prospect_horizontal_50_ec_av dropped) + 168 new; dropped seeds: valentine_distribution. Self-test: seed-study summary_tiefree.csv (plain) reproduced byte-identically: True; seed-study summary_tiefree.csv (md5) reproduced byte-identically: True; layer1_plain_scores/summary.csv reproduced byte-identically: True; layer1_plain_scores/report.md reproduced byte-identically: True; layer1nan_plain_scores/summary.csv reproduced byte-identically: True; layer1nan_plain_scores/report.md reproduced byte-identically: True

| family | status | arms | configs (full 549) | not on 549 (n = old381 + new168) | output |
|---|---|---|---|---|---|
| seed_plain | assembled | - | 90 (87) | datatype 369=295+74; extrema 401=260+141; sima_tpcdi 33=33+0 | results/plain549 |
| layer1_plain | assembled | melnik: assembled, fd: assembled | 178 (172) | datatype|fd 369=295+74; datatype|melnik 369=295+74; extrema|fd 401=260+141; extrema|melnik 402=261+141; sima_tpcdi|fd 33=33+0; sima_tpcdi|melnik 33=33+0 | layer1_plain549_scores |
| layer1nan_plain | assembled | melnik: assembled, fd: assembled, attach_id: assembled, coef_prod: assembled, coef_prod_fd: assembled, lit_none: assembled | 6 (6) | - | layer1nan_plain549_scores |
| layer1isr_plain | assembled | melnik: assembled, fd: assembled, attach_id: assembled, coef_prod: assembled, coef_prod_fd: assembled, lit_none: assembled | 12 (10) | isresmat_hybrid|lit_none 388=276+112; isresmat|lit_none 413=274+139 | layer1isr_plain549_scores |
| gvfull_plain | assembled | ucc_top1: assembled, ucc1_fd_topk: assembled | 179 (172) | datatype_nanfix|ucc1_fd_topk 381=381+0; datatype|ucc1_fd_topk 369=295+74; datatype|ucc_top1 369=295+74; extrema|ucc1_fd_topk 402=261+141; extrema|ucc_top1 402=261+141; sima_tpcdi|ucc1_fd_topk 33=33+0; sima_tpcdi|ucc_top1 33=33+0 | gvfull_plain549_scores |
| iter_plain | assembled | melnik: assembled, melnik_t1: assembled, melnik_t2: assembled, melnik_t3: assembled, melnik_t5: assembled, melnik_t10: assembled | 534 (521) | datatype|melnik 369=295+74; datatype|melnik_t1 369=295+74; datatype|melnik_t10 369=295+74; datatype|melnik_t2 369=295+74; datatype|melnik_t3 369=295+74; datatype|melnik_t5 369=295+74; extrema|melnik 402=261+141; sima_tpcdi|melnik 33=33+0; sima_tpcdi|melnik_t1 33=33+0; sima_tpcdi|melnik_t10 33=33+0; sima_tpcdi|melnik_t2 33=33+0; sima_tpcdi|melnik_t3 33=33+0; sima_tpcdi|melnik_t5 33=33+0 | iter_plain549_scores |
| parallel_plain | assembled | fd_split: assembled | 89 (86) | datatype|fd_split 369=295+74; extrema|fd_split 401=260+141; sima_tpcdi|fd_split 33=33+0 | parallel_plain549_scores |
| seed_md5 | assembled | - | 87 (86) | datatype 369=295+74 | results/md549 |
| layer1_md5 | assembled | melnik: assembled, fd: assembled | 174 (172) | datatype|fd 369=295+74; datatype|melnik 369=295+74 | layer1_md549_scores |
| layer1nan_md5 | assembled | melnik: assembled, fd: assembled, attach_id: assembled, coef_prod: assembled, coef_prod_fd: assembled, lit_none: assembled | 6 (6) | - | layer1nan_md549_scores |
| layer1isr_md5 | assembled | melnik: assembled, fd: assembled, attach_id: assembled, coef_prod: assembled, coef_prod_fd: assembled, lit_none: assembled | 12 (10) | isresmat_hybrid|lit_none 419=291+128; isresmat|lit_none 543=380+163 | layer1isr_md549_scores |
| gvfull_md5 | assembled | ucc_top1: assembled | 87 (86) | datatype|ucc_top1 369=295+74 | gvfull_md549_scores |
| iter_md5 | assembled | melnik: assembled, melnik_t1: assembled, melnik_t2: assembled, melnik_t3: assembled, melnik_t5: assembled, melnik_t10: assembled | 522 (516) | datatype|melnik 369=295+74; datatype|melnik_t1 369=295+74; datatype|melnik_t10 369=295+74; datatype|melnik_t2 369=295+74; datatype|melnik_t3 369=295+74; datatype|melnik_t5 369=295+74 | iter_md549_scores |
| parallel_md5 | assembled | fd_split: assembled | 87 (86) | datatype|fd_split 369=295+74 | parallel_md549_scores |

Pending sources:

- xt, wp, mt, cx (x81_<c>, wide_<c>, iter_x81_<c>, x54/x55 FD arms): *fix_<corpus> batches (non-Valentine corpora; they replace the FD arms in place and are not assembled here)

Flags on assembled rows (by seed):

- isresmat: not reproducible: GPU training without a fixed seed (matrices differ by up to 0.033 per cell between runs)
- isresmat_hybrid: not reproducible: GPU training without a fixed seed (matrices differ by up to 0.033 per cell between runs)

Check 1 exception (PRISMA on fixed FDs, rema fill: no old row to compare with), internal consistency:

- prismafix_m382fix_scores: 382 (seed, scenario) with identical seed columns across 24 arms
- prismafix_p382fix_scores: 382 (seed, scenario) with identical seed columns across 24 arms
- remafill_m382fix_scores: 258 (seed, scenario) with identical seed columns across 24 arms
- remafill_p382fix_scores: 258 (seed, scenario) with identical seed columns across 24 arms
- PRISMA: rows only from prismafix_* (381) and prismafs_* (168); its old rows (old FD profiles; md5 copies equal to plain by a feature-cache bug) are never used. rema/rema_shuffled: old or *fix rows for the 253 old scenarios, remafill_* for the 129 the old copy lacked, *fs_* on the 168.

Kept on the 382 by decision:

- tpass (timing pass): measured on the 382 scenarios as a separate descriptive cost measurement; not rerun on 549 (the user's decision 2026-11-02; the paper states that timings were measured on the 382)

Notes:

- seed_plain: FILE seeds on the 168 (learned_lofo, rema, rema_shuffled (possibly prisma, semstruct, semstruct_lm_only, sima_tpcdi)) from stdfs_pnew; datatype_nanfix: 381 from std_p382fix (seed columns checked against layer1nan_plain), 168 from stdnan_pnew; datatype_nanfix is not in gen_tables.META, so no paper panel uses its seed-study rows
- layer1_plain: arms attach_id, coef_prod, coef_prod_fd, lit_none are not rerun on 549 (long version only); FILE seeds on the 168 from layer1fs_pnew
- layer1nan_plain: datatype_nanfix: 381 fd from l1fd_p382fix (confirmed), coef_prod_fd from l1nanfix_p382fix, melnik, attach_id, coef_prod, lit_none keep the old rows; 168 from layer1nan_pnew
- layer1isr_plain: isresmat/isresmat_hybrid: 381 fd from l1fdisr_p382fix, coef_prod_fd from l1cpfdisr_p382fix (log 2026-10-31 18:20), the non-FD arms keep the old rows; 168 from layer1isr_pnew (GPU host)
- gvfull_plain: report.md (explore_cluster_batches.py) and deproles must be regenerated from the assembled rows
- iter_plain: fd_t* are not rerun (long version only); report.md (explore_iterations.py) to be regenerated
- parallel_plain: contrasts/report (score_parallel.py) to be regenerated
- seed_md5: FILE seeds on the 168 (learned_lofo, rema, rema_shuffled (possibly prisma, semstruct, semstruct_lm_only, sima_tpcdi)) from stdfs_mnew; datatype_nanfix: 381 from std_m382fix (seed columns checked against layer1nan_md5), 168 from stdnan_mnew; datatype_nanfix is not in gen_tables.META, so no paper panel uses its seed-study rows
- layer1_md5: arms attach_id, coef_prod, coef_prod_fd, lit_none are not rerun on 549 (long version only); FILE seeds on the 168 from layer1fs_mnew
- layer1nan_md5: datatype_nanfix: 381 fd from l1fd_m382fix (confirmed), coef_prod_fd from l1nanfix_m382fix, melnik, attach_id, coef_prod, lit_none keep the old rows; 168 from layer1nan_mnew
- layer1isr_md5: isresmat/isresmat_hybrid: 381 fd from l1fdisr_m382fix, coef_prod_fd from l1cpfdisr_m382fix (log 2026-10-31 18:20), the non-FD arms keep the old rows; 168 from layer1isr_mnew (GPU host)
- gvfull_md5: report.md (explore_cluster_batches.py) and deproles must be regenerated from the assembled rows
- iter_md5: fd_t* are not rerun (long version only); report.md (explore_iterations.py) to be regenerated
- parallel_md5: contrasts/report (score_parallel.py) to be regenerated
- plain: seed columns of layer-1 melnik rows on the 168 vs the seed study's new rows: 14663 compared, 0 differ
- md5: seed columns of layer-1 melnik rows on the 168 vs the seed study's new rows: 14354 compared, 0 differ
- seed_plain: seeds only in the std batch (no old seed-study row; both column sets from the std batch, seed columns checked against the old layer-1 rows): datatype_nanfix
- seed_plain: prismafix_p382fix_scores assembled (381 rows, seeds prisma)
- seed_plain: remafill_p382fix_scores assembled (258 rows, seeds rema, rema_shuffled)
- seed_plain: stdfs_pnew_scores assembled (840 rows, seeds learned_lofo, rema, rema_shuffled, semstruct, semstruct_lm_only)
- seed_plain: stdnan_pnew_scores assembled (168 rows, seeds datatype_nanfix)
- seed_plain: prismafs_pnew_scores assembled (168 rows, seeds prisma)
- seed_md5: seeds only in the std batch (no old seed-study row; both column sets from the std batch, seed columns checked against the old layer-1 rows): datatype_nanfix
- seed_md5: prismafix_m382fix_scores assembled (381 rows, seeds prisma)
- seed_md5: remafill_m382fix_scores assembled (258 rows, seeds rema, rema_shuffled)
- seed_md5: stdfs_mnew_scores assembled (672 rows, seeds rema, rema_shuffled, semstruct, semstruct_lm_only)
- seed_md5: stdnan_mnew_scores assembled (168 rows, seeds datatype_nanfix)
- seed_md5: prismafs_mnew_scores assembled (168 rows, seeds prisma)
