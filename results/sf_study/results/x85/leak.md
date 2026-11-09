# X85 follow-up: position leak of SF arms under X79's K1 column permutation (EXPLORATORY)

K1 output realigned to the base labels vs the base output; per arm over all matchers and bases.

| arm | bases × matchers | max abs difference | cells pairs with difference > 1e-9 | unreadable |
|---|---|---|---|---|
| melnik | 2490 | 2.60e-08 | 137 | 0 |
| fd | 2490 | 2.90e-08 | 137 | 0 |
| ucc_top1 | 2490 | 4.25e-02 | 167 | 0 |
| ucc1_fd_topk | 2490 | 4.28e-02 | 167 | 0 |

Largest differences:

| arm | matcher | base | max diff |
|---|---|---|---|
| ucc1_fd_topk | kang_naughton | imdb_joinable_s20261006_plain | 4.283e-02 |
| ucc1_fd_topk | jsd_values | imdb_semantically_joinable_s20261006_plain | 4.282e-02 |
| ucc1_fd_topk | kang_naughton | imdb_semantically_joinable_s20261006_plain | 4.276e-02 |
| ucc_top1 | jsd_values | imdb_semantically_joinable_s20261006_plain | 4.246e-02 |
| ucc_top1 | kang_naughton | imdb_joinable_s20261006_plain | 4.224e-02 |
| ucc_top1 | kang_naughton | imdb_semantically_joinable_s20261006_plain | 4.223e-02 |
| ucc1_fd_topk | jsd_values | imdb_joinable_s20261006_plain | 4.218e-02 |
| ucc1_fd_topk | containment | imdb_semantically_joinable_s20261006_plain | 4.202e-02 |
| ucc_top1 | containment | imdb_joinable_s20261006_plain | 4.189e-02 |
| ucc_top1 | jsd_values | imdb_joinable_s20261006_plain | 4.163e-02 |
| ucc_top1 | containment | imdb_semantically_joinable_s20261006_plain | 4.146e-02 |
| ucc1_fd_topk | containment | imdb_joinable_s20261006_plain | 3.980e-02 |
