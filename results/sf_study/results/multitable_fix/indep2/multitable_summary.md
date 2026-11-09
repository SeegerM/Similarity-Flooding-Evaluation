# X54/X55 (indep2): multi-table corpora (EXPLORATORY)

J_tf; per scenario the mean over included matchers; 95% bootstrap over scenarios within cell.

## Lahman (27 scenarios, 72 matchers)

| quantity | J_tf |
|---|---|
| q | +0.5321 [+0.5135, +0.5515] |
| SF as published | +0.5784 [+0.5542, +0.6037] |
| SF as published − q | +0.0463 [+0.0397, +0.0531] |
| SF with FD edges − q | -0.0296 [-0.0388, -0.0203] |
| SF top-1 key − q | +0.0472 [+0.0407, +0.0538] |
| A_d − SF as published | -0.0338 [-0.0398, -0.0277] |
| A_dt − SF as published | +0.0424 [+0.0372, +0.0477] |
| A_role − SF as published | +0.0439 [+0.0392, +0.0491] |
| A_role − SF top-1 key | +0.0431 [+0.0381, +0.0483] |
| A_rt − SF as published | +0.0873 [+0.0775, +0.0978] |
| A_dt − A_d | +0.0762 [+0.0658, +0.0870] |
| X52 composite (frozen; one matcher) | 0.7700 |
| composite − SF as published (panel mean) | +0.1916 |

## MusicBrainz (27 scenarios, 72 matchers)

| quantity | J_tf |
|---|---|
| q | +0.4150 [+0.4025, +0.4275] |
| SF as published | +0.5339 [+0.5030, +0.5638] |
| SF as published − q | +0.1189 [+0.0945, +0.1420] |
| SF with FD edges − q | +0.0754 [+0.0444, +0.1043] |
| SF top-1 key − q | +0.1157 [+0.0910, +0.1380] |
| A_d − SF as published | -0.0924 [-0.1157, -0.0679] |
| A_dt − SF as published | +0.0205 [+0.0176, +0.0233] |
| A_role − SF as published | +0.0058 [-0.0058, +0.0178] |
| A_role − SF top-1 key | +0.0090 [-0.0022, +0.0204] |
| A_rt − SF as published | +0.0663 [+0.0610, +0.0715] |
| A_dt − A_d | +0.1129 [+0.0863, +0.1380] |
| X52 composite (frozen; one matcher) | 0.6907 |
| composite − SF as published (panel mean) | +0.1568 |

## macro over Lahman and MusicBrainz

| contrast | J_tf |
|---|---|
| SF as published − q | +0.0826 [+0.0699, +0.0945] |
| SF with FD edges − q | +0.0229 [+0.0067, +0.0381] |
| SF top-1 key − q | +0.0814 [+0.0686, +0.0931] |
| A_d − SF as published | -0.0631 [-0.0749, -0.0505] |
| A_dt − SF as published | +0.0315 [+0.0284, +0.0345] |
| A_role − SF as published | +0.0249 [+0.0186, +0.0315] |
| A_role − SF top-1 key | +0.0260 [+0.0198, +0.0324] |
| A_rt − SF as published | +0.0768 [+0.0712, +0.0826] |
| A_dt − A_d | +0.0945 [+0.0803, +0.1081] |
