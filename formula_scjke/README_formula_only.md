# Formula-only SCJKE

This module is a non-ML DTI evidence model for DrugBAN cluster evaluation.

Design constraints:

- No neural network.
- No logistic regression or classifier fitting.
- No learned embeddings.
- Source DTI labels are only used to separate positive and negative evidence pools.

Core computation:

1. Protein sequence similarity:

`Kp = multi-scale k-mer cosine + amino-acid-composition cosine`

2. Drug similarity:

`Kd = fixed Morgan/MACCS/descriptor similarity`

3. Joint evidence:

`Kj = Kp * Kd`

4. Positive/negative evidence:

`E1 = similarity-weighted evidence from Y=1 source pairs`

`E0 = similarity-weighted evidence from Y=0 source pairs`

5. Output probability:

`p = pi1*E1/(pi1*E1 + pi0*E0)`

All parameters are fixed formulas; no optimization is performed.
