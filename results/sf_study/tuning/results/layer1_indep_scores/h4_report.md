# H4 (RQ4): SF effect of `melnik` on the independent corpus against Valentine plain

73 non-control initial matchers with full coverage in both corpora; families with >= 5 scenarios (plain: ChEMBL, Magellan, OpenData, TPC-DI, 380 scenarios; independent: AdventureWorks, Chinook, TPCH, 81 scenarios). J_tf = (ΔR_exp + ΔF1_Hr)/2, SF(melnik) − initial matrix.

| | J_tf plain | J_tf independent | difference | 95% CI | matchers with difference > 0 | verdict |
|---|---|---|---|---|---|---|
| H4 | +0.0211 | +0.0660 | +0.0449 | [+0.0379, +0.0516] | 66 of 73 | **confirmed** |

Bootstrap: 10000 stratified resamples per corpus, seed 20261002, corpora resampled independently.

## Per initial matcher

| matcher | plain | independent | difference |
|---|---|---|---|
| role_uncertainty | +0.1074 | +0.0600 | -0.0474 |
| topk_uniform_levenshtein | +0.0262 | +0.0019 | -0.0242 |
| kang_naughton | +0.1244 | +0.1032 | -0.0212 |
| probabilistic_levenshtein | +0.0224 | +0.0019 | -0.0205 |
| composite_dominants | +0.0104 | +0.0021 | -0.0083 |
| composite_average | +0.0088 | +0.0028 | -0.0060 |
| composite_confidence | +0.0060 | +0.0032 | -0.0028 |
| sbert_e5 | +0.0281 | +0.0320 | +0.0039 |
| sbert_mpnet_values | +0.0052 | +0.0106 | +0.0054 |
| valentine_distribution | +0.0061 | +0.0155 | +0.0093 |
| null_percentage | +0.1311 | +0.1411 | +0.0100 |
| unary_statistics | +0.0058 | +0.0166 | +0.0108 |
| char_distribution | -0.0198 | -0.0078 | +0.0120 |
| kang_naughton_profile | +0.0808 | +0.0932 | +0.0123 |
| role_gcov | +0.0936 | +0.1105 | +0.0169 |
| bert_base | +0.0146 | +0.0324 | +0.0177 |
| pmf | +0.0988 | +0.1177 | +0.0189 |
| minhash_containment | +0.0114 | +0.0337 | +0.0224 |
| semint | -0.0140 | +0.0086 | +0.0227 |
| magneto | -0.0104 | +0.0134 | +0.0238 |
| role_gcov_sig | +0.0928 | +0.1173 | +0.0245 |
| coma_instances | -0.0067 | +0.0193 | +0.0260 |
| cupid | -0.0042 | +0.0234 | +0.0276 |
| char3_soft | -0.0033 | +0.0263 | +0.0295 |
| unionability_pair | +0.0405 | +0.0708 | +0.0302 |
| token_soft | -0.0081 | +0.0252 | +0.0333 |
| minhash_jaccard | -0.0090 | +0.0282 | +0.0372 |
| unary_statistics_invariant | +0.0685 | +0.1060 | +0.0375 |
| fuzzy_jaccard | -0.0103 | +0.0283 | +0.0386 |
| valentine_jaccard | -0.0195 | +0.0209 | +0.0404 |
| sbert_mpnet | +0.0167 | +0.0573 | +0.0407 |
| containment | +0.0119 | +0.0560 | +0.0441 |
| wordnet_res | +0.0168 | +0.0619 | +0.0451 |
| sbert_minilm | +0.0082 | +0.0539 | +0.0458 |
| needleman_wunsch | +0.0105 | +0.0565 | +0.0460 |
| norms | +0.0142 | +0.0609 | +0.0467 |
| jsd_values | -0.0437 | +0.0045 | +0.0482 |
| qgram2_dice | +0.0041 | +0.0533 | +0.0491 |
| unionability | +0.0091 | +0.0584 | +0.0493 |
| jaccard_exact | -0.0207 | +0.0293 | +0.0500 |
| jaro_winkler | +0.0128 | +0.0644 | +0.0516 |
| monge_elkan_jw_expand | +0.0068 | +0.0606 | +0.0538 |
| cosine | +0.0076 | +0.0628 | +0.0552 |
| lcs | +0.0131 | +0.0685 | +0.0554 |
| distribution_shape_z | +0.0613 | +0.1180 | +0.0568 |
| wasserstein | +0.0331 | +0.0901 | +0.0570 |
| monge_elkan_porter | +0.0126 | +0.0700 | +0.0574 |
| qgram3_jaccard | -0.0007 | +0.0568 | +0.0575 |
| monge_elkan_jw | +0.0071 | +0.0646 | +0.0575 |
| wordnet_path | +0.0218 | +0.0811 | +0.0593 |
| smith_waterman_gotoh | +0.0135 | +0.0731 | +0.0596 |
| thesaurus_generic | +0.0066 | +0.0679 | +0.0613 |
| wordnet_wup | +0.0109 | +0.0730 | +0.0621 |
| wordnet_lch | +0.0145 | +0.0785 | +0.0640 |
| ks | +0.0178 | +0.0837 | +0.0659 |
| uniqueness | +0.1161 | +0.1826 | +0.0665 |
| levenshtein | +0.0096 | +0.0762 | +0.0666 |
| smith_waterman | +0.0075 | +0.0765 | +0.0690 |
| pattern_charclass | +0.0324 | +0.1027 | +0.0703 |
| distribution_shape | +0.0505 | +0.1220 | +0.0715 |
| wordnet_jcn | +0.0135 | +0.0850 | +0.0716 |
| pattern_shape | +0.0161 | +0.0881 | +0.0719 |
| wordnet_lin | +0.0156 | +0.0924 | +0.0768 |
| prefix_suffix_tfidf | +0.0143 | +0.0939 | +0.0796 |
| smoa | -0.0069 | +0.0734 | +0.0803 |
| composite_max | +0.0703 | +0.1543 | +0.0839 |
| soft_tfidf | -0.0091 | +0.0771 | +0.0862 |
| prefix_suffix | +0.0134 | +0.1005 | +0.0871 |
| coma | -0.0164 | +0.0716 | +0.0880 |
| hamming | +0.0190 | +0.1089 | +0.0899 |
| double_metaphone | -0.0100 | +0.1116 | +0.1217 |
| soundex | -0.0109 | +0.1150 | +0.1259 |
| distinct_count | +0.0749 | +0.2247 | +0.1497 |
