#!/bin/bash
# 2026-10-31 ~21:45: SF arms for the new FILE seeds of REQUESTS_FROM_A.md item 2, cloned from rema's configurations.
#   prismafix_{p,m}382fix  PRISMA on fixed-HyFD profiles, all 24 arms rema/prisma had on the 382, fixed-FD corpora
#   remafill_{p,m}382fix   rema, rema_shuffled fill, same 24 arms, only the 5 datasets the old rema run lacks
#   prismafs_{pnew,mnew}   PRISMA on the 168 new, the arms of the other FILE seeds there (stdfs/layer1fs/...)
set -e
cd "$(dirname "$0")/../../sf_study"   # the study folder
src382() {   # $1 = plain|md5, $2 = seed label in the source stages
  local r=$1 s=$2 fix; [ $r = plain ] && fix=p382fix || fix=m382fix
  echo std_$fix/${s}__std
  for a in melnik fd attach_id coef_prod coef_prod_fd lit_none; do echo layer1_$r/${s}__$a; done
  for a in ucc_top1 ucc1_fd_topk ucc_complete_top1 fd_const005; do echo gvfull_$r/${s}__$a; done
  for t in 1 2 3 5 10; do echo iter_$r/${s}__melnik_t$t iter_$r/${s}__fd_t$t; done
  for a in fd_split fd_pavg fd_pmax; do echo parallel_$r/${s}__$a; done
}
srcnew() {   # $1 = pnew|mnew
  local r=$1
  echo stdfs_$r/rema__std layer1fs_$r/rema__melnik layer1fs_$r/rema__fd gvfullfs_$r/rema__ucc_top1
  [ $r = pnew ] && echo gvfullfs_$r/rema__ucc1_fd_topk
  for t in 1 2 3 5 10; do echo iterfs_$r/rema__melnik_t$t; done
  echo parallelfs_$r/rema__fd_split
}
FILL=TPC-DI-Semantically-Joinable,TPC-DI-Unionable,TPC-DI-View-Unionable,Wikidata-Joinable,Wikidata-Semantically-Joinable
python servers/reseed_stage.py --suffix prismafix_p382fix --seed prisma --regime plain --corpus p382fix \
    --seed-dir reference/seeds/plain_fdfix/prisma --src $(src382 plain rema)
python servers/reseed_stage.py --suffix prismafix_m382fix --seed prisma --regime md5 --corpus m382fix \
    --seed-dir reference/seeds/md5_fdfix/prisma --src $(src382 md5 rema)
for r in plain md5; do
  [ $r = plain ] && fix=p382fix || fix=m382fix
  python servers/reseed_stage.py --suffix remafill_$fix --seed rema --regime $r --corpus $fix --datasets $FILL \
      --seed-dir reference/seeds/${r}_fill/rema --src $(src382 $r rema)
  # rema_shuffled into a second stage (labels must differ from rema's), merged below
  python servers/reseed_stage.py --suffix remafillshuf_$fix --seed rema_shuffled --regime $r --corpus $fix --datasets $FILL \
      --seed-dir reference/seeds/${r}_fill/rema_shuffled --src $(src382 $r rema_shuffled)
  mv tuning/configs/remafillshuf_$fix/* tuning/configs/remafill_$fix/ && rmdir tuning/configs/remafillshuf_$fix
  sed -i "s#tuning/results/remafillshuf_$fix/#tuning/results/remafill_$fix/#" tuning/configs/remafill_$fix/rema_shuffled__*/general.yaml
done
for r in pnew mnew; do
  python servers/reseed_stage.py --suffix prismafs_$r --seed prisma --regime $r --corpus $r \
      --seed-dir reference/seeds/$r/prisma --src $(srcnew $r)
done
