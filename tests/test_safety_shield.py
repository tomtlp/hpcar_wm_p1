from src.recovery_actions import RecoveryAction
from src.safety_shield import SafetyShield


def test_safety_shield_prevents_pump_on_at_low_level():
    shield = SafetyShield()
    belief = {
        "level_est": 10.0,
        "trust_LIT101": 1.0,
        "trust_FIT101": 1.0,
        "trust_MV101": 1.0,
        "trust_P101": 1.0,
        "trust_P102": 1.0,
    }
    state = {"p101_state": 1, "p102_state": 0, "p101_command": 1, "p102_command": 0, "mv101_state": 1}
    decision = shield.filter_action(RecoveryAction.R3_SWITCH_TO_BACKUP_PUMP, belief, state)
    assert decision.intervened
    assert decision.action != RecoveryAction.R3_SWITCH_TO_BACKUP_PUMP
