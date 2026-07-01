from src.p1_simulator import P1Simulator


def test_level_increases_when_mv_open_and_pumps_off():
    sim = P1Simulator({"process_noise_std": 0.0, "sensor_noise_std": 0.0}, seed=1)
    start = sim.state.level_true
    next_state = sim.step_physics_only(mv101_state=1, p101_state=0, p102_state=0)
    assert next_state.level_true > start


def test_level_decreases_when_pumps_on_and_mv_closed():
    sim = P1Simulator({"process_noise_std": 0.0, "sensor_noise_std": 0.0}, seed=1)
    start = sim.state.level_true
    next_state = sim.step_physics_only(mv101_state=0, p101_state=1, p102_state=0)
    assert next_state.level_true < start
