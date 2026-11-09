# X50 (indep): extraction with abstention and many-to-many matches (EXPLORATORY)

73 initial matchers, 81 scenarios; families: AdventureWorks, Chinook, TPCH. Policies chosen leave-one-family-out per matrix type:

- q: AdventureWorks rel@1.0; Chinook forced; TPCH forced; _other forced
- A_d: AdventureWorks rel@1.0; Chinook hung@0.3; TPCH hung@0.3; _other hung@0.3
- sf: AdventureWorks forced; Chinook forced; TPCH forced; _other forced

| | forced (one-to-one) F1 | validated policy F1 |
|---|---|---|
| initial matcher q | 0.6400 [0.6357, 0.6444] | 0.6256 [0.6215, 0.6298] |
| additive datatype term | 0.6759 [0.6716, 0.6804] | 0.6654 [0.6614, 0.6694] |
| SF as published | 0.7289 [0.7208, 0.7366] | 0.7289 [0.7210, 0.7367] |

| contrast | forced | validated policy |
|---|---|---|
| SF − q | +0.0889 [+0.0826, +0.0951] | +0.1033 [+0.0965, +0.1097] |
| A_d − SF | -0.0530 [-0.0594, -0.0464] | -0.0636 [-0.0701, -0.0569] |
| validated − forced, q | -0.0144 [-0.0157, -0.0131] | |
| validated − forced, SF | +0.0000 [+0.0000, +0.0000] | |
