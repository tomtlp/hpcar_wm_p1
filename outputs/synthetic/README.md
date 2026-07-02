# synthetic 输出分类说明

本目录是 synthetic 多 seed 实验结果，当前运行 seeds 为 `0,1,2,3,4`。

模型文件不放在本目录，已移动到：

```text
/mnt/data01/tlp/model/world_model_synthetic.pt
```

## 目录结构

| 目录 | 内容 | 优先看 |
| --- | --- | --- |
| `00_summary_metrics/` | 汇总指标、逐攻击指标、最佳方法、trust 检测和 world model 评估 | `results_summary.csv`, `metrics_by_method_attack.csv`, `world_model_eval.csv` |
| `01_timeseries/` | 所有方法/攻击/seed 的逐时间步轨迹 | `per_run_timeseries.csv` |
| `02_plots/` | synthetic 结果图 | `level_trajectories.png`, `action_timeline.png`, `safety_violations_bar.png`, `production_loss_bar.png` |

## 压缩包

`synthetic_outputs_by_category.zip` 是按当前分类目录重新打包的结果，已排除模型文件。
