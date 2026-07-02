# real_swat 输出分类说明

本目录只整理真实 SWaT 结果中的 CSV/PNG 文件，并在本目录内提供压缩包。模型文件不在这里，模型已移动到 `/mnt/data01/tlp/model`。

## 目录结构

| 目录 | 内容 | 优先看 |
| --- | --- | --- |
| `00_preprocess_calibration/` | 文件扫描、列映射、标签分布、攻击窗口、P1 校准和执行器映射 | `swat_file_inventory.csv`, `swat_attack_windows.csv`, `swat_p1_calibration_report.csv` |
| `01_p1_log_eval/` | P1 真实日志诊断、窗口指标、trust 检测、预测误差 | `real_swat_p1_log_eval_summary_valid_windows.csv`, `real_swat_p1_detection_by_window_valid.csv`, `real_swat_p1_world_model_eval.csv` |
| `02_p1_plots/` | P1 日志诊断图、窗口覆盖图、残差图、trust 图、根因分数图 | `real_swat_p1_valid_windows_overlay.png`, `real_swat_p1_root_cause_scores.png` |
| `03_counterfactual/` | 反事实恢复 rollout、候选动作评分、动作选择和 B5 消融 | `real_swat_p1_counterfactual_summary_valid_windows.csv`, `real_swat_p1_counterfactual_candidate_scores.csv` |
| `04_case_studies/` | 典型攻击窗口 case study 的摘要、rollout 和图 | `case_study_attack_*_summary.csv`, `case_study_attack_*_plot.png` |
| `05_hybrid/` | 真实 normal 校准仿真的主场景恢复结果、轨迹、图和动作效果调试 | `real_swat_hybrid_summary.csv`, `real_swat_hybrid_metrics_by_method_attack.csv` |
| `06_hybrid_stress/` | stress 场景的恢复结果、轨迹、图和动作效果调试 | `real_swat_hybrid_stress_summary.csv`, `real_swat_hybrid_stress_metrics_by_method_attack.csv` |

## 已删除的大文件

为节省空间，三份约 500M 的真实日志逐时间步明细没有保留：

- `real_swat_p1_timeseries.csv`
- `real_swat_prediction_timeseries.csv`
- `real_swat_trust_timeseries.csv`

保留的 `real_swat_hybrid_timeseries.csv` 和 `real_swat_hybrid_stress_timeseries.csv` 是仿真轨迹明细，体积较小。

## 压缩包

本目录内的 `real_swat_outputs_by_category.zip` 是按当前分类目录重新打包的结果。压缩包排除了模型文件和其他 zip 文件。
