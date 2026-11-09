# X50 (md5): extraction with abstention and many-to-many matches (EXPLORATORY)

78 initial matchers, 549 scenarios; families: ChEMBL, Magellan, OpenData, TPC-DI. Policies chosen leave-one-family-out per matrix type:

- q: ChEMBL forced; Magellan rel@1.0; OpenData forced; TPC-DI rel@0.9; _other forced
- A_d: ChEMBL hung@0.9; Magellan rel@1.0; OpenData hung@0.3; TPC-DI rel@0.95; _other hung@0.3
- sf: ChEMBL rel@0.95; Magellan rel@1.0; OpenData rel@0.95; TPC-DI rel@0.95; _other rel@0.95

| | forced (one-to-one) F1 | validated policy F1 |
|---|---|---|
| initial matcher q | 0.1197 [0.1090, 0.1305] | 0.1004 [0.0873, 0.1148] |
| additive datatype term | 0.1346 [0.1227, 0.1461] | 0.1172 [0.1025, 0.1327] |
| SF as published | 0.1329 [0.1214, 0.1443] | 0.1170 [0.1021, 0.1318] |

| contrast | forced | validated policy |
|---|---|---|
| SF − q | +0.0133 [+0.0120, +0.0145] | +0.0166 [+0.0148, +0.0184] |
| A_d − SF | +0.0017 [+0.0012, +0.0022] | +0.0003 [-0.0005, +0.0011] |
| validated − forced, q | -0.0193 [-0.0236, -0.0145] | |
| validated − forced, SF | -0.0160 [-0.0202, -0.0112] | |
