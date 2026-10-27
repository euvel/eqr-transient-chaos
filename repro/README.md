# float64 reproduction (CPU, ~10 s)

`python repro_eqr_float64.py` needs `loopscape` (`pip install "loopscape @ git+https://github.com/GilpinLab/loopscape@16b52aa1e9382f8cdd7af45a1ba7eb1efa755b3d"`).
It prints the ratio of separations after one loop for two initial offsets (1e-6 vs 1e-9); float64 gives ~1000:

```
get_solver(dtype=float64)                      forward_dtype=torch.float32  d(1e-6)/d(1e-9) =    1.21
+ config_overrides forward_dtype="float64"     forward_dtype=torch.float64  d(1e-6)/d(1e-9) =    1.80
+ rms_norm patched to keep float64             forward_dtype=torch.float64  d(1e-6)/d(1e-9) = 1000.00
```

Reported upstream as [GilpinLab/loopscape#1](https://github.com/GilpinLab/loopscape/issues/1).
