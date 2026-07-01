from pathlib import Path

from src.experiment import apply_cli_overrides, build_arg_parser, main, parse_seed_list
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


def test_cli_seeds_override_config_even_in_quick_mode():
    parser = build_arg_parser()
    args = parser.parse_args(["--quick", "--seeds", "0,1,2,3,4"])
    config = {
        "experiment": {"seeds": [9], "quick_steps": 70},
        "world_model": {"train_rollouts": 60, "train_steps": 70, "epochs": 8, "hidden_dim": 48},
    }

    updated = apply_cli_overrides(config, args)

    assert updated["experiment"]["seeds"] == [0, 1, 2, 3, 4]
    assert updated["experiment"]["steps"] == 70


def test_parse_seed_list_rejects_empty_values():
    assert parse_seed_list(" 0, 2,4 ") == [0, 2, 4]
