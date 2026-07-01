# 面向 SWaT P1 的危险优先因果-POMDP 可行动恢复世界模型

本仓库是一个用于水处理工业控制系统攻击后安全恢复的最小可行研究原型。当前版本只聚焦 SWaT Stage P1，不是完整 SWaT 系统复刻。

这个 MVP 用来验证以下核心流程：

1. 基于因果/规则的诊断；
2. 感知信任状态的局部回滚；
3. 动作条件化世界模型预测；
4. 危险优先的恢复动作规划；
5. 在执行恢复动作前使用安全屏蔽器避免二次事故。

## 环境安装

推荐使用 Python 3.10 或更高版本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

本项目只使用本地 Python 依赖，不需要外部服务，不连接真实 PLC，也不需要联网运行实验。

## 运行合成数据 MVP

快速模式适合先确认流程是否跑通，通常在 CPU 上 1 分钟内完成：

```powershell
python -m src.experiment --config configs/default.yaml --mode synthetic --quick
```

完整合成模式默认运行 5 个随机种子：

```powershell
python -m src.experiment --config configs/default.yaml --mode synthetic
```

实验结果会写入 `outputs/`：

- `results_summary.csv`
- `results_summary_all.csv`
- `results_summary_attack_only.csv`
- `per_run_timeseries.csv`
- `metrics_by_method_attack.csv`
- `per_attack_best_method_by_safety.csv`
- `per_attack_best_method_by_production.csv`
- `trust_detection_by_tag.csv`
- `world_model_eval.csv`
- `world_model.pt`
- `level_trajectories.png`
- `action_timeline.png`
- `safety_violations_bar.png`
- `production_loss_bar.png`
- `shield_interventions_bar.png`
- `trust_mask_example.png`

## 可选 SWaT CSV 模式

CSV 模式是尽力支持模式。程序会在给定 CSV 中大小写不敏感地搜索类似 `LIT101`、`FIT101`、`MV101`、`P101`、`P102` 的列。

如果缺少必要列，程序会打印警告并自动回退到合成仿真模式，不会崩溃。

```powershell
python -m src.experiment --config configs/default.yaml --mode swat_csv --csv_path path/to/file.csv --quick
```

当前版本对 SWaT CSV 的使用较轻量：主要用于列名检查和可选初始液位读取。如果 CSV 中 `LIT101` 已经处于本 MVP 使用的 `0-100` 液位范围内，程序会尝试用它初始化仿真。

## 真实 SWaT 离线数据模式

真实 SWaT 数据目录默认期望为：

```text
dataset/SWat/
```

程序会递归发现其中的 `.csv`、`.xlsx`、`.xls` 文件，自动识别 normal 文件、attack 文件和 attack list 文件，并生成：

- `swat_file_inventory.csv`
- `swat_column_mapping.json`
- `swat_label_profile.csv`
- `swat_attack_windows.csv`
- `swat_actuator_mapping.csv`
- `swat_preprocess_report.csv`
- `swat_p1_calibration.json`
- `swat_p1_calibration_report.csv`

重要限制：真实 SWaT CSV/XLSX 是离线日志，恢复动作无法改变日志中已经记录好的未来轨迹。因此，真实数据模式分成三类，不能把 `log_eval` 解释成真实闭环恢复。

### 真实 SWaT 日志诊断评估

```powershell
python -m src.experiment --config configs/default.yaml --mode real_swat --swat_dir dataset/SWat --real_swat_task log_eval --quick
```

`log_eval` 只验证：

- 攻击检测；
- 根因/信任掩码推断；
- LIT101 状态重构；
- world model 预测；
- attack window 或 label 下的离线检测指标。

主要输出：

- `real_swat_log_eval_summary.csv`
- `real_swat_detection_metrics.csv`
- `real_swat_trust_detection_by_tag.csv`
- `real_swat_world_model_eval.csv`
- `real_swat_timeseries.csv`
- `real_swat_trust_timeseries.csv`
- `real_swat_prediction_timeseries.csv`
- `real_swat_level_prediction.png`
- `real_swat_trust_mask.png`
- `real_swat_residuals.png`
- `real_swat_attack_windows_overlay.png`

这些输出的 `evaluation_type` 为 `offline_log_diagnosis`。

### 真实 SWaT 反事实恢复评估

```powershell
python -m src.experiment --config configs/default.yaml --mode real_swat --swat_dir dataset/SWat --real_swat_task counterfactual --quick
```

`counterfactual` 从真实攻击窗口取攻击前上下文，用学习到的世界模型或校准后的 P1 动力学模型滚动预测候选恢复动作后果。它评估的是“模型中的反事实恢复决策”，不是对真实日志的闭环控制。

主要输出：

- `real_swat_counterfactual_summary.csv`
- `real_swat_counterfactual_by_attack.csv`
- `real_swat_counterfactual_action_timeline.csv`
- `real_swat_counterfactual_level_rollouts.png`
- `real_swat_counterfactual_actions.png`

这些输出的 `evaluation_type` 为 `counterfactual_model_rollout`。

### 真实校准混合仿真评估

```powershell
python -m src.experiment --config configs/default.yaml --mode real_swat --swat_dir dataset/SWat --real_swat_task hybrid --quick
```

`hybrid` 会先用真实 SWaT normal 数据校准 P1 简化仿真器，然后在这个真实统计特性约束下的仿真器里重放合成攻击并运行 B1-B5 恢复策略。这是用真实 normal 行为校准后的 simulation-in-the-loop 恢复验证。

主要输出：

- `real_swat_hybrid_summary.csv`
- `real_swat_hybrid_metrics_by_method_attack.csv`
- `real_swat_hybrid_level_trajectories.png`
- `real_swat_hybrid_production_loss_bar.png`
- `real_swat_hybrid_safety_violations_bar.png`

这些输出的 `evaluation_type` 为 `real_calibrated_simulation`。

## 真实 SWaT 结果如何解释

- `log_eval`：验证诊断、trust mask、状态重构和预测能力；不要声称闭环恢复效果。
- `counterfactual`：验证基于学习动力学的恢复动作选择是否合理；结果属于模型反事实 rollout。
- `hybrid`：验证恢复 controller 在“由 SWaT normal 数据校准的仿真器”中的表现；结果属于真实校准仿真。

如果真实数据目录不存在、P1 列缺失或标签缺失，程序会尽量继续运行可运行部分；无法继续时会写出 `real_swat_error.txt`，不会影响 synthetic 模式。

## POMDP 简化建模

仿真器明确区分真实物理状态和传感器观测：

- `level_true` 表示 T101 水箱真实液位；
- `lit101_obs` 表示 LIT101 观测值，它可能被攻击篡改。

这个分离是把攻击后恢复问题视为 POMDP 的关键原因：攻击发生后，控制器看到的传感器值不一定等于真实物理状态。

为了让 MVP 简洁可运行，本项目不实现完整 POMDP 求解器，而是通过信任状态重构得到一个 belief state：

```text
[level_est, fit_est, actuator states, trust mask, hazard priority, attack belief]
```

也就是说，系统先判断哪些变量可信，再用可信观测和物理估计重构决策状态。

## 因果图与信任感知局部回滚

P1 的因果图在代码中手工编码：

```text
MV101 -> FIT101
P101/P102 -> outflow
FIT101 and outflow -> LIT101_true_next
LIT101_true -> LIT101_obs
LIT101_obs -> PLC logic -> actuator commands
attack nodes -> corrupted observations or actuator states
```

诊断规则会比较：

- 基于质量守恒估计的液位与 `LIT101` 观测值；
- `FIT101` 与 `MV101` 状态的一致性；
- 执行器命令与反馈状态；
- PLC 阈值控制逻辑是否被违反。

当某个变量被判定为不可信时，系统只替换这个变量，而不是完全丢弃所有观测：

```text
reconstructed = trusted observation + untrusted physics estimate
```

例如，当 `LIT101` 不可信时，恢复控制和规划会使用质量守恒估计的液位，而不是直接相信 `lit101_obs`。

## 恢复动作

规划器选择的是高层恢复动作，而不是任意底层泵阀组合：

- `R0_KEEP_CURRENT`
- `R1_ISOLATE_LIT101_USE_ESTIMATED_LEVEL`
- `R2_FREEZE_MV101_SAFE`
- `R3_SWITCH_TO_BACKUP_PUMP`
- `R4_LIMIT_PUMP_SWITCHING`
- `R5_P1_FALLBACK_CONTROL`
- `R6_BLOCK_SCADA_REMOTE_WRITE`
- `R7_LOCAL_SAFE_SHUTDOWN`
- `R8_GRADUAL_RERAMP`
- `R9_EMERGENCY_DRAIN_BOTH_PUMPS`
- `R10_SENSOR_ISOLATION_AND_FALLBACK`

这些动作会在 `src/recovery_actions.py` 中转换为具体的 `MV101`、`P101`、`P102` 控制命令。

其中 `R9` 用于高液位且 `MV101` 可能 stuck open 的场景：如果至少一个泵可信，就优先用可信泵紧急排水；如果 `P101` 被强制关闭但 `P102` 可信，则使用 `P102`。`R10` 用于隔离不可信 `LIT101` 并切换到基于重构液位的本地 fallback 控制。

## 对比方法与 proposed 方法

实验比较以下方法：

- `B1_FULL_SHUTDOWN`：攻击检测后直接全停机；
- `B2_RULE_BASED_FALLBACK`：使用简单阈值 fallback 控制；
- `B3_ANOMALY_PRIORITY_RECOVERY`：只根据最大异常残差选择动作；
- `B4_WORLD_MODEL_NO_TRUST`：使用世界模型，但直接相信传感器观测；
- `B5_PROPOSED`：因果/规则 trust mask + 局部回滚 + 危险优先规划 + safety shield。

核心假设是：相比全停机、简单规则 fallback、异常优先恢复、以及不考虑信任状态的世界模型恢复，`B5_PROPOSED` 应该在 P1 攻击场景下减少安全违规时长、生产损失和不安全恢复动作。

## 攻击场景

默认实验包含：

- `normal`
- `LIT101_FDI`
- `LIT101_DRIFT`
- `LIT101_REPLAY`
- `MV101_STUCK_OPEN`
- `MV101_STUCK_CLOSED`
- `P101_FORCED_OFF`
- `COMBINED_LIT101_FDI_MV101_OPEN`

## 评价指标

实验会计算：

- `time_to_safe_set`：回到安全集合的时间；
- `time_to_safe_after_attack`：攻击开始后首次进入安全集合的时间；
- `time_to_target_after_attack`：攻击开始后首次进入目标液位区间的时间；
- `time_to_recover_after_first_violation`：首次安全违规后，连续保持安全 N 步所需时间；
- `recovery_success`：有限时域内是否成功恢复；
- `safety_violation_duration`：安全约束违规持续时间；
- `max_level_overshoot`：超过安全上界的最大幅度；
- `max_level_undershoot`：低于安全下界的最大幅度；
- `pump_empty_run_count`：低液位下泵空转次数；
- `production_loss`：相对正常基线的产水损失；
- `action_cost`：恢复动作代价；
- `false_recovery_count`：正常场景下误触发恢复次数；
- `shield_intervention_count`：安全屏蔽器干预次数；
- `trust_mask_accuracy`：在有攻击真值时的 trust mask 准确率；
- `trust_detection_precision`、`trust_detection_recall`、`trust_detection_f1`：按 compromised 变量检测的精确率、召回率和 F1；
- `one_step_rmse`、`multi_step_rollout_rmse`、`attack_period_rmse`：世界模型预测误差；
- `raw_observation_rmse`、`full_rollback_rmse`、`partial_rollback_rmse`、`trust_aware_reconstruction_rmse`：观测值、全回滚估计、局部回滚估计和信任感知重构的液位误差。

## 运行测试

```powershell
pytest
```

测试覆盖：

- 仿真器液位动力学；
- LIT101 攻击是否只改变观测、不改变真实液位；
- 大幅 FDI 下 trust mask 是否能标记 `LIT101` 不可信；
- 安全屏蔽器是否阻止低液位泵启动；
- 规划器是否返回合法恢复动作；
- quick 实验是否能端到端运行并写出 `results_summary.csv`。

## 当前原型边界

这个代码不是精确的 SWaT 数字孪生，也不是真实工控恢复控制器。它是一个研究脚手架，用来快速验证：

```text
攻击诊断 -> 信任状态重构 -> 世界模型预测 -> 危险优先规划 -> 安全屏蔽执行
```

后续可以继续扩展到真实 SWaT CSV 深度接入、P1-P3 多阶段建模、offline RL/MPC 对比方法，以及更丰富的攻击图和因果传播分析。
