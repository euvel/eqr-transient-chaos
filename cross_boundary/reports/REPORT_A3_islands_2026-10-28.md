# A3: islands vs stripes in 2-D slices (toys, numpy, 2026-10-28): FAILED
Script `a3_islands.py` (prediction in the docstring), raw `a3_islands.json`. Runtime 6 s.

| depth L | ROUGH islands (median; per slice) | FILAM islands (median; per slice) |
|---|---|---|
| 0.5 | 27.5; [0, 16, 61, 103, 39, 15] | 1; [0, 0, 1, 1, 1, 1] |
| 0.05 | 0.5; [0, 32, 66, 1, 0, 0] | 0.5; [0, 0, 2, 6, 1, 0] |
| 0.005 | 0; [0, 96, 53, 0, 0, 0] | 0; [0, 0, 1, 0, 0, 0] |
| 0.0005 | 0; [0, 45, 188, 0, 0, 0] | 1; [4, 0, 57, 0, 2, 0] |
Prediction (ROUGH >= 3 islands at every depth; FILAM <= 1 at small depths) fails: ROUGH has islands at fine scales in
only 2 of 6 slices, and FILAM shows 57 in one.

Why: the "contour loops of a rough surface" picture assumes isotropic roughness. In a skew system the fine-scale
roughness of W comes from the FASTEST churn direction alone (at scale r its terms have amplitude r^(ln a/ln 3) and beat
the slower x2 cascade, r^(ln a/ln 2)). So W is effectively one-dimensional at fine scales, and its level curves in a slice are
nearly straight stripes, the same as filamentary sheets. The degeneracy reappears in topology: at fine scales both
boundaries look like a Cantor set of stripes. Islands appear only in slices where two cascades compete (2/6), which is
not a usable test.
