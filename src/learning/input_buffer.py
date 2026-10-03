"""Online input synchronization for the Lepid-IO network pipeline.

The paper network consumes a uniformly sampled window containing calibrated
IMU data, measured wing displacement, and measured wing phase.  This module is
owned by :mod:`learning` so that constructing the network input is independent
of any downstream state estimator.
"""

from __future__ import annotations

import numpy as np


def interpolate_wrapped_phase(last_phase, phase, alpha):
    """Interpolate phases along the shortest arc and wrap them to ``[0, 2π)``."""
    last_phase = np.asarray(last_phase, dtype=float)
    phase = np.asarray(phase, dtype=float)
    delta = (phase - last_phase + np.pi) % (2.0 * np.pi) - np.pi
    return (last_phase + float(alpha) * delta) % (2.0 * np.pi)


def _linear_interpolate(last_value, value, alpha):
    """Interpolate two column-vector samples without a SciPy dependency."""
    last_value = np.asarray(last_value, dtype=float).T
    value = np.asarray(value, dtype=float).T
    return last_value + float(alpha) * (value - last_value)


class NetInputBuffer:
    """Uniformly sampled, ten-channel raw input history.

    The stored channels are IMU(6), measured wing displacement(2), and raw
    measured phase(2).  Phase sin/cos encoding is intentionally performed by
    the model, so the buffer contract stays identical to the final CSV schema.
    """

    def __init__(self):
        self.net_t_us = np.array([])
        self.net_gyr = np.array([])
        self.net_displacement = np.array([])
        self.net_phase = np.array([])
        self.net_acc = np.array([])

    def add_data_interpolated(
        self,
        last_t_us,
        t_us,
        last_gyr,
        gyr,
        last_displacement,
        displacement,
        last_phase,
        phase,
        last_acc,
        acc,
        requested_interpolated_t_us,
    ):
        """Interpolate one sample at the requested network-grid timestamp."""
        assert isinstance(last_t_us, int)
        assert isinstance(t_us, int)

        if last_t_us < 0:
            gyr_interp = gyr.T
            displacement_interp = displacement.T
            phase_interp = phase.T
            acc_interp = acc.T
        else:
            try:
                alpha = (
                    (requested_interpolated_t_us - last_t_us)
                    / float(t_us - last_t_us)
                )
                gyr_interp = _linear_interpolate(last_gyr, gyr, alpha)
                phase_interp = interpolate_wrapped_phase(
                    last_phase.T, phase.T, alpha
                )
                displacement_interp = _linear_interpolate(
                    last_displacement, displacement, alpha
                )
                acc_interp = _linear_interpolate(last_acc, acc, alpha)
            except ValueError as error:
                raise ValueError(
                    "cannot interpolate network input at "
                    f"{requested_interpolated_t_us} between {last_t_us} and {t_us}"
                ) from error

        self._add_data(
            requested_interpolated_t_us,
            gyr_interp,
            displacement_interp,
            phase_interp,
            acc_interp,
        )

    def _add_data(self, t_us, gyr, displacement, phase, acc):
        assert isinstance(t_us, int)
        if len(self.net_t_us) > 0:
            assert t_us > self.net_t_us[-1], (
                f"timestamp {t_us} must be later than existing timestamp "
                f"{self.net_t_us[-1]}"
            )
        self.net_t_us = np.append(self.net_t_us, t_us)
        self.net_gyr = np.append(self.net_gyr, gyr).reshape(-1, 3)
        self.net_displacement = np.append(
            self.net_displacement, displacement
        ).reshape(-1, 2)
        self.net_phase = np.append(self.net_phase, phase).reshape(-1, 2)
        self.net_acc = np.append(self.net_acc, acc).reshape(-1, 3)

    def get_data_from_to(self, t_begin_us: int, t_us_end: int):
        """Return a closed time interval in model feature order."""
        assert isinstance(t_begin_us, int)
        assert isinstance(t_us_end, int)
        begin_idx = np.where(self.net_t_us == t_begin_us)[0][0]
        end_idx = np.where(self.net_t_us == t_us_end)[0][0]
        return (
            self.net_gyr[begin_idx:end_idx + 1, :],
            self.net_displacement[begin_idx:end_idx + 1, :],
            self.net_phase[begin_idx:end_idx + 1, :],
            self.net_acc[begin_idx:end_idx + 1, :],
            self.net_t_us[begin_idx:end_idx + 1],
        )

    def throw_data_before(self, t_begin_us: int):
        """Discard history strictly older than ``t_begin_us``."""
        assert isinstance(t_begin_us, int)
        begin_idx = np.where(self.net_t_us == t_begin_us)[0][0]
        self.net_gyr = self.net_gyr[begin_idx:, :]
        self.net_displacement = self.net_displacement[begin_idx:, :]
        self.net_phase = self.net_phase[begin_idx:, :]
        self.net_acc = self.net_acc[begin_idx:, :]
        self.net_t_us = self.net_t_us[begin_idx:]

    def total_net_data(self):
        return self.net_t_us.shape[0]
