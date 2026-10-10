# Live-A / Frozen-A 边界诊断结果

全部预定 seed；相关性为描述性分析，不作因果或显著性声明。

| 数据集 | 方法 | Final | AAA | Forgetting |
|---|---|---:|---:|---:|
| inr | live | 79.56±0.10 | 82.56±0.04 | 4.88±0.79 |
| inr | frozen | 78.50±0.04 | 81.90±0.09 | 5.04±0.66 |
| c100 | live | 87.77±0.61 | 91.90±0.53 | 7.24±1.02 |
| c100 | frozen | 87.84±0.55 | 91.93±0.40 | 6.52±1.06 |
| cub | live | 80.65±1.56 | 87.77±0.89 | 11.10±0.31 |
| cub | frozen | 83.93±0.27 | 88.84±0.47 | 7.11±0.89 |

## 可恢复性与旧类变化

每项为 Live-A 的 seed×任务边界 Spearman 相关；逐层值不作为独立样本。

| 数据集 | Seed | 残差 | 旧类损失 | ρ |
|---|---:|---|---|---:|
| cifar224 | 汇总 | r_rec | old_drop_hls | -0.386 |
| cifar224 | 汇总 | r_rec | old_drop_deploy | -0.332 |
| cifar224 | 汇总 | r_weighted | old_drop_hls | -0.253 |
| cifar224 | 汇总 | r_weighted | old_drop_deploy | -0.161 |
| cifar224 | 1993 | r_rec | old_drop_hls | -0.483 |
| cifar224 | 1993 | r_rec | old_drop_deploy | -0.583 |
| cifar224 | 1993 | r_weighted | old_drop_hls | -0.217 |
| cifar224 | 1993 | r_weighted | old_drop_deploy | 0.033 |
| cifar224 | 1994 | r_rec | old_drop_hls | -0.293 |
| cifar224 | 1994 | r_rec | old_drop_deploy | -0.200 |
| cifar224 | 1994 | r_weighted | old_drop_hls | -0.050 |
| cifar224 | 1994 | r_weighted | old_drop_deploy | 0.133 |
| cifar224 | 1995 | r_rec | old_drop_hls | -0.333 |
| cifar224 | 1995 | r_rec | old_drop_deploy | -0.176 |
| cifar224 | 1995 | r_weighted | old_drop_hls | -0.250 |
| cifar224 | 1995 | r_weighted | old_drop_deploy | -0.343 |
| cub | 汇总 | r_rec | old_drop_hls | 0.337 |
| cub | 汇总 | r_rec | old_drop_deploy | 0.413 |
| cub | 汇总 | r_weighted | old_drop_hls | 0.308 |
| cub | 汇总 | r_weighted | old_drop_deploy | 0.273 |
| cub | 1 | r_rec | old_drop_hls | 0.017 |
| cub | 1 | r_rec | old_drop_deploy | 0.283 |
| cub | 1 | r_weighted | old_drop_hls | 0.033 |
| cub | 1 | r_weighted | old_drop_deploy | 0.333 |
| cub | 2 | r_rec | old_drop_hls | -0.167 |
| cub | 2 | r_rec | old_drop_deploy | 0.500 |
| cub | 2 | r_weighted | old_drop_hls | 0.000 |
| cub | 2 | r_weighted | old_drop_deploy | -0.500 |
| cub | 3 | r_rec | old_drop_hls | 0.617 |
| cub | 3 | r_rec | old_drop_deploy | 0.400 |
| cub | 3 | r_weighted | old_drop_hls | 0.567 |
| cub | 3 | r_weighted | old_drop_deploy | 0.483 |
| imagenetr | 汇总 | r_rec | old_drop_hls | -0.109 |
| imagenetr | 汇总 | r_rec | old_drop_deploy | -0.255 |
| imagenetr | 汇总 | r_weighted | old_drop_hls | -0.390 |
| imagenetr | 汇总 | r_weighted | old_drop_deploy | -0.092 |
| imagenetr | 1995 | r_rec | old_drop_hls | -0.617 |
| imagenetr | 1995 | r_rec | old_drop_deploy | -0.333 |
| imagenetr | 1995 | r_weighted | old_drop_hls | -0.683 |
| imagenetr | 1995 | r_weighted | old_drop_deploy | 0.000 |
| imagenetr | 1996 | r_rec | old_drop_hls | -0.033 |
| imagenetr | 1996 | r_rec | old_drop_deploy | -0.567 |
| imagenetr | 1996 | r_weighted | old_drop_hls | -0.517 |
| imagenetr | 1996 | r_weighted | old_drop_deploy | -0.400 |
| imagenetr | 1997 | r_rec | old_drop_hls | -0.117 |
| imagenetr | 1997 | r_rec | old_drop_deploy | -0.283 |
| imagenetr | 1997 | r_weighted | old_drop_hls | -0.017 |
| imagenetr | 1997 | r_weighted | old_drop_deploy | -0.200 |

逐任务阶段指标见 `boundary_rows.csv`；Live−Frozen 配对差值见 `paired_boundaries.csv`。
