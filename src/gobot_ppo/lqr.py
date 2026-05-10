"""LQR controller helpers for the CartPole smoke task.

This module intentionally stays pure Python and independent from Gobot's C++
simulation layer. Gobot only sees generic joint effort commands.
"""

import math


DEFAULT_CARTPOLE_LQR_GAIN = (
    -17.41564600376392,
    -22.394686026248433,
    -132.93399597485794,
    -16.969100804277314,
)


class CartPoleLQRController:
    """Continuous CartPole stabilizer for [x, x_dot, theta, theta_dot]."""

    def __init__(
        self,
        gain=DEFAULT_CARTPOLE_LQR_GAIN,
        target_position=1.0,
        force_limit=20.0,
        theta_reference=0.0,
        theta_sign=1.0,
    ):
        if len(gain) != 4:
            raise ValueError("CartPole LQR gain must contain 4 values")
        self.gain = tuple(float(value) for value in gain)
        self.target_position = float(target_position)
        self.force_limit = float(force_limit)
        self.theta_reference = float(theta_reference)
        self.theta_sign = 1.0 if float(theta_sign) >= 0.0 else -1.0
        self.last_effort = 0.0
        self.last_action = 0.0

    def reset(self):
        self.last_effort = 0.0
        self.last_action = 0.0

    def effort(self, observation, target_position=None):
        x, x_dot, theta, theta_dot, target_error = _cartpole_state(observation)
        target = float(target_position) if target_position is not None else x + target_error
        if not math.isfinite(target):
            target = self.target_position

        state = (
            x - target,
            x_dot,
            self.theta_sign * (theta - self.theta_reference),
            self.theta_sign * theta_dot,
        )
        value = -sum(gain * state_value for gain, state_value in zip(self.gain, state))
        value = _clamp(value, -self.force_limit, self.force_limit)
        self.last_effort = value
        self.last_action = value / self.force_limit if self.force_limit > 0.0 else 0.0
        return value

    def action(self, observation, target_position=None):
        self.effort(observation, target_position=target_position)
        return [self.last_action]


def cartpole_lqr_action(
    observation,
    target_position=None,
    force_limit=20.0,
    gain=DEFAULT_CARTPOLE_LQR_GAIN,
    theta_sign=1.0,
):
    controller = CartPoleLQRController(
        gain=gain,
        target_position=1.0 if target_position is None else target_position,
        force_limit=force_limit,
        theta_sign=theta_sign,
    )
    return controller.action(observation, target_position=target_position)


def _cartpole_state(observation):
    values = [float(value) for value in observation]
    if len(values) < 5:
        raise ValueError(f"expected CartPole observation size 5, got {len(values)}")
    return tuple(0.0 if not math.isfinite(value) else value for value in values[:5])


def _clamp(value, lower, upper):
    return max(float(lower), min(float(upper), float(value)))
