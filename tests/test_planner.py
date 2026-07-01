from pathlib import Path

from src.experiment import main
from src.p1_simulator import P1Simulator
from src.planner import HazardPrioritizedPlanner
from src.recovery_actions import RecoveryAction
from src.safety_shield import SafetyShield
from src.world_model import ActionConditionedWorldModel


def test_planner_returns_valid_action_enum():
    sim = P1Simulator({"process_noise_std": 0.0, "sensor_noise_std": 0.0}, seed=4)
    wm = ActionConditionedWorldModel({"enabled": False})
    planner = HazardPrioritizedPlanner(wm, SafetyShield())
    decision = planner.select_action(sim.state.to_dict(), [])
    assert isinstance(decision.action, RecoveryAction)


def test_experiment_quick_runs_end_to_end(tmp_path):
    config = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    result = main([
        "--config",
        str(config),
        "--mode",
        "synthetic",
        "--quick",
        "--output_dir",
        str(tmp_path),
    ])
    assert (tmp_path / "results_summary.csv").exists()
    assert result["summary"].exists()
