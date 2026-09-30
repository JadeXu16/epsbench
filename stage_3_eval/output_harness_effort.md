| model | harness | effort | PDE (%) | paired ΔPDE, pp (90% CI) | coverage (%) | tools | input (K) | output (K) |
|---|---|---|---|---|---|---|---|---|
| Opus 4.6 | OpenCode | default | 0.113 | -- | 69.8% (447/447) | 35 | 1056 | 9.7 |
| Opus 4.6 | Claude Code | default | 0.117 | +0.001 [-0.005, +0.004] | 64.7% (447/447) | 34 | 1029 | 36.4 |
| Opus 4.6 | OpenCode | max | 0.106 | -0.002 [-0.006, +0.001] | 65.2% (419/447) | 34 | 1011 | 40.5 |
| Opus 4.6 | Claude Code | max | 0.114 | -0.004 [-0.008, +0.002] | 65.3% (447/447) | 34 | 1032 | 40.4 |
| GPT-5.5 | OpenCode | default | 0.131 | -- | 61.3% (447/447) | 47 | 837 | 6.2 |
| GPT-5.5 | Vanilla | default | 0.110 | -0.003 [-0.007, +0.002] | 53.5% (447/447) | 38 | 561 | 8.6 |
| GPT-5.5 | OpenCode | max | 0.135 | -0.000 [-0.003, +0.003] | 53.5% (447/447) | 48 | 821 | 12.6 |

Each configuration against analyst consensus (median paired ΔPDE, pp):
  Opus 4.6  OpenCode     default  -0.0020 [-0.0110, +0.0058]  win 75/149
  Opus 4.6  Claude Code  default  -0.0008 [-0.0097, +0.0050]  win 75/149
  Opus 4.6  OpenCode     max      -0.0046 [-0.0139, +0.0037]  win 81/149
  Opus 4.6  Claude Code  max      -0.0016 [-0.0086, +0.0038]  win 77/149
  GPT-5.5   OpenCode     default  +0.0002 [-0.0047, +0.0094]  win 74/149
  GPT-5.5   Vanilla      default  -0.0000 [-0.0075, +0.0068]  win 75/149
  GPT-5.5   OpenCode     max      +0.0018 [-0.0058, +0.0134]  win 71/149
