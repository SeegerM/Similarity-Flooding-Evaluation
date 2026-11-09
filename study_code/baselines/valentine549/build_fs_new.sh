#!/bin/bash
# 2026-10-31: SF arms on the 168 new Valentine scenarios for the FILE seeds the peer session produced
# (REQUESTS_FROM_A.md item 1). Same arms as the regular seeds got on pnew/mnew:
#   stdfs (std), layer1fs (melnik, fd), gvfullfs (ucc_top1 [+ ucc1_fd_topk on pnew]), iterfs (melnik_t1..t10),
#   parallelfs (fd_split). Each regime has its own corpus (pnew, mnew).
set -e
cd "$(dirname "$0")/../../sf_study"   # the study folder
for r in pnew mnew; do
  if [ $r = pnew ]; then S=learned_lofo,rema,rema_shuffled,semstruct,semstruct_lm_only; else S=rema,rema_shuffled,semstruct,semstruct_lm_only; fi
  python reference/make_std_configs.py --corpus $r --regime $r --arm std --file-seeds $S --suffix stdfs_$r
  python reference/make_std_configs.py --corpus $r --regime $r --arm melnik --file-seeds $S --suffix layer1fs_$r
  python reference/make_std_configs.py --corpus $r --regime $r --arm fd --file-seeds $S --suffix layer1fs_$r --append
  python servers/clone_stage.py --base layer1fs_$r --base-arm melnik --suffix gvfullfs_$r --arm ucc_top1 \
      --yaml uccFilterThreshold=0.0 dependencyTopK=1
  if [ $r = pnew ]; then
    python servers/clone_stage.py --base layer1fs_$r --base-arm fd --suffix gvfullfs_$r --arm ucc1_fd_topk --append \
        --yaml uccFilterThreshold=0.0 constraintWeighting=TOPK_HYBRID \
        --props smx.sf.topK=1 smx.sf.uccTopK=1 smx.sf.uccUnweighted=true smx.cp.overlay=overlay_gvb
  fi
  first=1
  for t in 1 2 3 5 10; do
    ap=""; [ $first = 1 ] || ap=--append; first=0
    python servers/clone_stage.py --base layer1fs_$r --base-arm melnik --suffix iterfs_$r --arm melnik_t$t $ap \
        --props smx.sf.maxIterations=$((t-1))
  done
  python servers/clone_stage.py --base layer1fs_$r --base-arm fd --suffix parallelfs_$r --arm fd_split \
      --props smx.sf.parallelEdges=SPLIT
done
