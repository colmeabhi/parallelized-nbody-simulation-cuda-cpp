| N | version | ms per step | speedup | interactions / s | GFLOP/s |
|---|---|--:|--:|--:|--:|
| 1K | CPU, 1 thread | 0.377 | 1.0x | 2.8 billion | 56 |
| 1K | CPU, OpenMP | 0.081 | 4.7x | 13.0 billion | 260 |
| 1K | GPU, naive | 0.124 | 3.1x | 8.5 billion | 170 |
| 1K | GPU, tiled | 0.056 | 6.7x | 18.7 billion | 374 |
| 4K | CPU, 1 thread | 5.815 | 1.0x | 2.9 billion | 58 |
| 4K | CPU, OpenMP | 0.826 | 7.0x | 20.3 billion | 406 |
| 4K | GPU, naive | 0.489 | 11.9x | 34.3 billion | 687 |
| 4K | GPU, tiled | 0.189 | 30.8x | 88.7 billion | 1,775 |
| 16K | CPU, 1 thread | 92.381 | 1.0x | 2.9 billion | 58 |
| 16K | CPU, OpenMP | 12.027 | 7.7x | 22.3 billion | 446 |
| 16K | GPU, naive | 1.834 | 50.4x | 146.4 billion | 2,927 |
| 16K | GPU, tiled | 0.741 | 124.7x | 362.2 billion | 7,245 |
| 64K | CPU, 1 thread | 1493.349 | 1.0x | 2.9 billion | 58 |
| 64K | CPU, OpenMP | 189.460 | 7.9x | 22.7 billion | 453 |
| 64K | GPU, naive | 9.061 | 164.8x | 474.0 billion | 9,480 |
| 64K | GPU, tiled | 7.973 | 187.3x | 538.7 billion | 10,774 |
