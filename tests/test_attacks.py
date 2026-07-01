from src.attacks import create_attack
from src.causal_logic import infer_trust_mask
from src.p1_simulator import P1Simulator
from src.recovery_actions import RecoveryAction


def test_lit101_fdi_changes_observation_not_true_level():
    sim = P1Simulator({"process_noise_std": 0.0, "sensor_noise_std": 0.0}, seed=2)
    attack = create_attack("LIT101_FDI", {"start_step": 0, "lit101_fdi_offset": 20.0})
    state, _ = sim.step(RecoveryAction.R0_KEEP_CURRENT, attack)
    assert state.lit101_obs == state.level_true + 20.0
    assert state.level_true != state.lit101_obs


def test_trust_mask_marks_lit101_untrusted_under_large_fdi():
    sim = P1Simulator({"process_noise_std": 0.0, "sensor_noise_std": 0.0}, seed=3)
    prev = sim.state.to_dict()
    attack = create_attack("LIT101_FDI", {"start_step": 0, "lit101_fdi_offset": 30.0})
    state, _ = sim.step(RecoveryAction.R0_KEEP_CURRENT, attack)
    trust = infer_trust_mask(
        state.to_dict(),
        [prev],
        {"lit101_residual_threshold": 5.0, "sensor_noise_std": 0.0},
    )
    assert trust["LIT101"] == 0
