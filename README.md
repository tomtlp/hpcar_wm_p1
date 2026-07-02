# 面向 SWaT P1 的危险优先因果-POMDP 可行动恢复世界模型

本仓库是一个用于水处理工业控制系统攻击后安全恢复的研究原型。当前版本只聚焦 SWaT Stage P1，不是完整 SWaT 系统复刻，也不是真实工控控制器。

核心流程：

1. 基于因果/规则的攻击诊断；
2. trust mask 和局部状态回滚；
3. 动作条件化世界模型预测；
4. 危险优先恢复动作规划；
5. safety shield 执行前拦截不安全动作。

## 当前文件布局

为了避免 `/home` 根分区被大文件占满，本机当前把数据、模型和大结果放在 `/mnt/data01/tlp`。

| 类型 | 路径 | 说明 |
| --- | --- | --- |
| 数据集实际目录 | `/mnt/data01/tlp/dataset` | SWaT 原始数据和 CSV 数据 |
| 项目内数据入口 | `dataset -> /mnt/data01/tlp/dataset` | 保持 `dataset/SWat/...` 路径可用 |
| 模型目录 | `/mnt/data01/tlp/model` | 不放进 `outputs/` |
| synthetic 结果 | `outputs/synthetic` | 已按类别分目录整理 |
| real SWaT 结果入口 | `outputs/real_swat -> /mnt/data01/tlp/outputs/real_swat` | 已按类别分目录整理 |
| real SWaT 分类压缩包 | `outputs/real_swat/real_swat_outputs_by_category.zip` | 当前约 23M，已排除模型和 3 个大 timeseries |
| 完整 real SWaT 结果备份 | `/mnt/data01/tlp/hpcar_real_swat_csv_full` | 原始运行结果目录，含 README |

当前模型文件：

- `/mnt/data01/tlp/model/world_model_synthetic.pt`
- `/mnt/data01/tlp/model/world_model_real_swat.json`

## src 代码结构

`src` 已按职责分成子包，同时保留根目录兼容 wrapper，所以旧路径仍可用，例如 `from src.planner import HazardPrioritizedPlanner` 和 `python -m src.experiment`。

| 目录 | 职责 | 代表文件 |
| --- | --- | --- |
| `src/core/` | P1 仿真器、攻击场景、恢复动作、安全屏蔽 | `p1_simulator.py`, `attacks.py`, `recovery_actions.py`, `safety_shield.py` |
| `src/diagnosis/` | 因果/规则诊断、trust mask、状态重构 | `causal_logic.py` |
| `src/planning/` | baseline 策略和危险优先规划器 | `baselines.py`, `planner.py` |
| `src/models/` | 世界模型和真实 SWaT 轻量模型 | `world_model.py` |
| `src/data/` | SWaT 文件发现、读取、预处理、窗口解析和校准 | `swat_loader.py`, `swat_preprocess.py`, `swat_attack_windows.py`, `swat_calibration.py`, `data_loader.py` |
| `src/evaluation/` | 指标和画图 | `metrics.py`, `plotting.py` |
| `src/runners/` | CLI 实验入口和真实 SWaT 任务编排 | `experiment.py`, `real_swat_experiment.py` |
| `src/common/` | 配置、路径、随机种子等工具函数 | `utils.py` |
| `src/*.py` | 兼容 wrapper | 保持旧 import 和 `python -m src.experiment` 可用 |

## 环境安装

推荐 Python 3.10。

Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

本项目只使用本地 Python 依赖，不连接真实 PLC，也不需要外部服务。

## 运行合成实验

快速检查：

```bash
.venv/bin/python -m src.experiment \
  --config configs/default.yaml \
  --mode synthetic \
  --quick \
  --output_dir outputs/synthetic
```

完整 synthetic：

```bash
.venv/bin/python -m src.experiment \
  --config configs/default.yaml \
  --mode synthetic \
  --output_dir outputs/synthetic
```

多 seed：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
.venv_gpu/bin/python -m src.experiment \
  --config configs/default.yaml \
  --mode synthetic \
  --seeds 0,1,2,3,4 \
  --output_dir outputs/synthetic
```

synthetic 的 CSV/PNG 结果整理在 `outputs/synthetic/`。模型文件已移动到 `/mnt/data01/tlp/model/world_model_synthetic.pt`。

当前 synthetic 输出分类：

| 目录/文件 | 内容 |
| --- | --- |
| `outputs/synthetic/00_summary_metrics/` | `results_summary*.csv`、`metrics_by_method_attack.csv`、最佳方法、trust 检测和 `world_model_eval.csv` |
| `outputs/synthetic/01_timeseries/` | `per_run_timeseries.csv` |
| `outputs/synthetic/02_plots/` | `level_trajectories.png`、`action_timeline.png`、安全/生产/屏蔽器图 |
| `outputs/synthetic/README.md` | synthetic 输出说明 |
| `outputs/synthetic/synthetic_outputs_by_category.zip` | 分类后的 synthetic 结果压缩包，已排除模型 |

## 运行真实 SWaT 离线实验

真实数据入口：

```text
dataset/SWat/
```

本机 `dataset` 是符号链接，实际数据在 `/mnt/data01/tlp/dataset`。当前 CSV 数据在：

```text
dataset/SWat/csv/normal.csv
dataset/SWat/csv/merged.csv
dataset/SWat/csv/attack.csv
```

推荐使用 CSV 版本，避免反复解析大 Excel。当前 full 结果已经使用：

- normal: `dataset/SWat/csv/normal.csv`
- attack: `dataset/SWat/csv/merged.csv`
- attack list: `dataset/SWat/List_of_attacks_Final.xlsx`

真实 SWaT 是离线日志，恢复动作不能改变日志中已经记录好的未来轨迹。因此真实数据实验分三种口径：

| 任务 | 含义 | 能否表示闭环恢复 |
| --- | --- | --- |
| `p1_log_eval` | P1 日志诊断、trust mask、重构和预测 | 否 |
| `counterfactual` | 在模型中评估候选恢复动作的反事实 rollout | 否，属于模型反事实 |
| `hybrid` | 用真实 normal 校准 P1 仿真器，再跑恢复 controller | 可以讨论恢复效果，但口径是仿真 |

快速跑 P1 日志诊断：

```bash
.venv/bin/python -m src.experiment \
  --config configs/default.yaml \
  --mode real_swat \
  --swat_dir dataset/SWat \
  --real_swat_task p1_log_eval \
  --quick \
  --output_dir outputs/real_swat
```

快速跑反事实恢复：

```bash
.venv/bin/python -m src.experiment \
  --config configs/default.yaml \
  --mode real_swat \
  --swat_dir dataset/SWat \
  --real_swat_task counterfactual \
  --quick \
  --output_dir outputs/real_swat
```

快速跑真实校准 hybrid：

```bash
.venv/bin/python -m src.experiment \
  --config configs/default.yaml \
  --mode real_swat \
  --swat_dir dataset/SWat \
  --real_swat_task hybrid \
  --quick \
  --output_dir outputs/real_swat
```

## 当前真实 SWaT 结果

当前整理后的 real SWaT CSV/PNG 结果可从项目内访问：

```text
outputs/real_swat
```

实际目录：

```text
/mnt/data01/tlp/outputs/real_swat
```

完整结果目录：

```text
/mnt/data01/tlp/hpcar_real_swat_csv_full
```

分类压缩包在 `outputs/real_swat` 目录内：

```text
outputs/real_swat/real_swat_outputs_by_category.zip
```

压缩包已排除：

- `*.pt`
- `*.pth`
- `*.pkl`
- `*.joblib`
- `*.ckpt`
- `world_model_real_swat.json`

为节省空间，当前结果也删除了三份约 500M 的真实日志逐时间步明细：

- `real_swat_p1_timeseries.csv`
- `real_swat_prediction_timeseries.csv`
- `real_swat_trust_timeseries.csv`

## real SWaT 输出怎么读

完整逐文件说明在：

```text
outputs/real_swat/README.md
```

优先看这些文件：

| 文件 | 用途 |
| --- | --- |
| `real_swat_p1_report.md` | P1 真实日志诊断文字报告 |
| `real_swat_p1_log_eval_summary_valid_windows.csv` | P1 有效窗口日志诊断主指标 |
| `real_swat_p1_detection_by_window_valid.csv` | 逐 P1 攻击窗口检测结果 |
| `real_swat_p1_world_model_eval.csv` | 真实日志上的预测误差 |
| `real_swat_p1_counterfactual_summary_valid_windows.csv` | 反事实恢复主结果 |
| `real_swat_p1_counterfactual_candidate_scores.csv` | 候选动作评分，解释为什么选择某个恢复动作 |
| `real_swat_hybrid_summary.csv` | 真实 normal 校准仿真的主恢复结果 |
| `real_swat_hybrid_metrics_by_method_attack.csv` | hybrid 逐方法、逐攻击详细指标 |
| `real_swat_hybrid_stress_summary.csv` | stress 场景主恢复结果 |
| `real_swat_hybrid_stress_metrics_by_method_attack.csv` | stress 逐方法、逐攻击详细指标 |

输出分类：

| 类别目录 | 文件模式 | 含义 |
| --- | --- | --- |
| `00_preprocess_calibration/` | `swat_*`, `real_swat_actuator_*` | 文件扫描、列映射、标签分布、攻击窗口、P1 边界和诊断阈值 |
| `01_p1_log_eval/` | `real_swat_p1_*log_eval*`, `real_swat_p1_detection*`, `real_swat_p1_trust*`, `real_swat_p1_world_model_eval.csv` | 真实日志中的检测、trust mask 和预测评估 |
| `02_p1_plots/` | `real_swat_p1_*.png`, `real_swat_level_prediction.png`, `real_swat_residuals.png`, `real_swat_trust_mask.png` | 阈值、攻击窗口、trust heatmap、根因分数和预测残差图 |
| `03_counterfactual/` | `real_swat_p1_counterfactual_*`, `real_swat_counterfactual_*` | 模型 rollout 下的候选动作评分、动作选择和预测轨迹 |
| `04_case_studies/` | `case_study_attack_*` | 典型攻击窗口的单独解释 |
| `05_hybrid/` | `real_swat_hybrid_*` | 真实 normal 校准仿真下的恢复指标、轨迹和图 |
| `06_hybrid_stress/` | `real_swat_hybrid_stress_*` | 更强攻击/故障条件下的恢复验证 |
| 模型文件 | `/mnt/data01/tlp/model/*` | 不放入 `outputs` 和分享压缩包 |

## 方法与建模说明

### POMDP 简化

仿真器区分真实物理状态和传感器观测：

- `level_true`：T101 水箱真实液位；
- `lit101_obs`：LIT101 观测值，可能被攻击篡改。

攻击后恢复问题被视为一个简化 POMDP。当前原型不实现完整 POMDP 求解器，而是通过 trust mask 和物理估计构造 belief state：

```text
[level_est, fit_est, actuator states, trust mask, hazard priority, attack belief]
```

当某个变量不可信时，只替换该变量，而不是丢弃所有观测：

```text
reconstructed = trusted observation + untrusted physics estimate
```

### P1 因果图

P1 因果关系在代码中手工编码：

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

### 恢复动作

规划器选择高层恢复动作，再由 `src/core/recovery_actions.py` 转换为 `MV101`、`P101`、`P102` 控制命令。`src/recovery_actions.py` 仍作为兼容 wrapper 保留。

常用动作：

- `R0_KEEP_CURRENT`
- `R1_ISOLATE_LIT101_USE_ESTIMATED_LEVEL`
- `R3_SWITCH_TO_BACKUP_PUMP`
- `R5_P1_FALLBACK_CONTROL`
- `R7_LOCAL_SAFE_SHUTDOWN`
- `R9_EMERGENCY_DRAIN_BOTH_PUMPS`
- `R10_SENSOR_ISOLATION_AND_FALLBACK`

其中 `R9` 用于高液位且 `MV101` 可能 stuck open 的场景；如果 `P101` 不可信但 `P102` 可信，则优先使用 `P102` 排水。`R10` 用于隔离不可信 `LIT101` 并切换到基于重构液位的 fallback 控制。

### 对比方法

| 方法 | 含义 |
| --- | --- |
| `B1_FULL_SHUTDOWN` | 检测到攻击后直接全停机 |
| `B2_RULE_BASED_FALLBACK` | 简单阈值 fallback 控制 |
| `B3_ANOMALY_PRIORITY_RECOVERY` | 只按最大异常残差选择恢复 |
| `B4_WORLD_MODEL_NO_TRUST` | 用世界模型但直接相信传感器观测 |
| `B5_PROPOSED` | trust mask + 局部回滚 + 危险优先规划 + safety shield |
| `B5_FULL` / `B5_NO_*` | hybrid 消融版本 |

核心假设：`B5_PROPOSED` 应减少安全违规时长、生产损失和不安全恢复动作。

## 评价指标

常用恢复指标：

- `time_to_safe_set`：回到安全集合的时间；
- `time_to_safe_after_attack`：攻击开始后首次进入安全集合的时间；
- `time_to_target_after_attack`：攻击开始后首次进入目标液位区间的时间；
- `recovery_success`：有限时域内是否成功恢复；
- `safety_violation_duration`：安全约束违规持续时间；
- `hard_safety_violation_duration`：硬安全约束违规持续时间；
- `max_level_overshoot` / `max_level_undershoot`：超过/低于安全边界的最大幅度；
- `pump_empty_run_count`：低液位下泵空转次数；
- `production` / `production_loss`：产水代理和生产损失；
- `action_cost`：恢复动作代价；
- `shield_intervention_count`：安全屏蔽器干预次数；
- `trust_mask_accuracy`、`trust_detection_precision`、`trust_detection_recall`、`trust_detection_f1`：trust mask 检测指标；
- `one_step_rmse`、`multi_step_rollout_rmse`：世界模型预测误差。

## 测试

```bash
.venv/bin/python -m pytest
```

测试覆盖：

- 仿真器液位动力学；
- LIT101 攻击只改变观测、不改变真实液位；
- trust mask 是否能标记不可信变量；
- safety shield 是否阻止低液位泵启动；
- 规划器是否合法返回恢复动作；
- CLI `--seeds` 覆盖；
- P1 target tag 归一化、攻击窗口排除、P1 标签生成和 normal 阈值校准；
- 真实校准 hybrid 是否使用真实单位边界，并产生非零生产代理指标。

## 当前原型边界

这个代码不是精确的 SWaT 数字孪生，也不是真实工控恢复控制器。它是一个研究脚手架，用来验证：

```text
攻击诊断 -> 信任状态重构 -> 世界模型预测 -> 危险优先规划 -> 安全屏蔽执行
```

后续可继续扩展到 P1-P3 多阶段建模、offline RL/MPC 对比、更多攻击图和因果传播分析。
