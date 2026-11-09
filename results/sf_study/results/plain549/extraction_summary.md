# X50 (plain): extraction with abstention and many-to-many matches (EXPLORATORY)

78 initial matchers, 549 scenarios; families: ChEMBL, Magellan, OpenData, TPC-DI. Policies chosen leave-one-family-out per matrix type:

- q: ChEMBL rel@1.0; Magellan rel@1.0; OpenData rel@1.0; TPC-DI forced; _other rel@1.0
- A_d: ChEMBL hung@0.8; Magellan hung@0.9; OpenData hung@0.3; TPC-DI hung@0.8; _other hung@0.8
- sf: ChEMBL rel@1.0; Magellan rel@1.0; OpenData rel@1.0; TPC-DI hung@0.3; _other rel@1.0

| | forced (one-to-one) F1 | validated policy F1 |
|---|---|---|
| initial matcher q | 0.5438 [0.5299, 0.5578] | 0.5437 [0.5294, 0.5582] |
| additive datatype term | 0.5719 [0.5566, 0.5870] | 0.5834 [0.5687, 0.5974] |
| SF as published | 0.5542 [0.5395, 0.5697] | 0.5590 [0.5438, 0.5743] |

| contrast | forced | validated policy |
|---|---|---|
| SF − q | +0.0104 [+0.0065, +0.0144] | +0.0153 [+0.0121, +0.0188] |
| A_d − SF | +0.0177 [+0.0162, +0.0191] | +0.0244 [+0.0182, +0.0305] |
| validated − forced, q | -0.0002 [-0.0055, +0.0052] | |
| validated − forced, SF | +0.0048 [-0.0010, +0.0106] | |
