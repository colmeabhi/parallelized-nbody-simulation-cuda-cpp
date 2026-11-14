| N | version | ms per step | speedup | interactions / s | GFLOP/s |
|---|---|--:|--:|--:|--:|
| 1K | CPU, 1 thread | 1.994 | 1.0x | 0.5 billion | 11 |
| 1K | CPU, OpenMP | 0.276 | 7.2x | 3.8 billion | 76 |
| 1K | GPU, naive | 0.122 | 16.3x | 8.6 billion | 171 |
| 1K | GPU, tiled | 0.059 | 33.9x | 17.9 billion | 357 |
| 4K | CPU, 1 thread | 31.665 | 1.0x | 0.5 billion | 11 |
| 4K | CPU, OpenMP | 4.048 | 7.8x | 4.1 billion | 83 |
| 4K | GPU, naive | 0.479 | 66.1x | 35.0 billion | 700 |
| 4K | GPU, tiled | 0.198 | 159.9x | 84.7 billion | 1,694 |
| 16K | CPU, 1 thread | 506.093 | 1.0x | 0.5 billion | 11 |
| 16K | CPU, OpenMP | 64.155 | 7.9x | 4.2 billion | 84 |
| 16K | GPU, naive | 1.802 | 280.8x | 149.0 billion | 2,979 |
| 16K | GPU, tiled | 0.761 | 665.2x | 352.8 billion | 7,056 |
| 64K | CPU, 1 thread | 8122.282 | 1.0x | 0.5 billion | 11 |
| 64K | CPU, OpenMP | 1017.815 | 8.0x | 4.2 billion | 84 |
| 64K | GPU, naive | 11.227 | 723.5x | 382.6 billion | 7,651 |
| 64K | GPU, tiled | 8.061 | 1007.6x | 532.8 billion | 10,657 |
