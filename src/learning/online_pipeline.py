"""Online synchronization and window extraction for Lepid-IO inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from learning.input_buffer import NetInputBuffer


@dataclass(frozen=True)
class NetworkWindow:
    """A synchronized input window in the final three-modality contract."""

    timestamps_s: np.ndarray
    gyr_body: np.ndarray
    acc_body: np.ndarray
    wing_displacement: np.ndarray
    wing_phase: np.ndarray


class OnlineInputPipeline:
    """Own the uniform input grid, buffer, and causal window boundaries.

    A deployment can feed raw
    sensor frames into the buffer and request the window ending at prediction
    time ``k`` regardless of how the predicted displacement is later consumed.
    """

    def __init__(self, sampling_freq: float, window_size: int, buffer=None):
        if sampling_freq <= 0:
            raise ValueError("sampling_freq must be positive")
        if int(window_size) < 2:
            raise ValueError("window_size must contain at least two samples")
        self.sampling_freq = float(sampling_freq)
        self.window_size = int(window_size)
        self.sample_period_us = int(round(1e6 / self.sampling_freq))
        self.buffer = NetInputBuffer() if buffer is None else buffer
        self.next_sample_t_us = None

    def start_at(self, timestamp_us: int):
        """Anchor the uniform network grid at the first source timestamp."""
        self.next_sample_t_us = int(timestamp_us)

    def append_available(
        self,
        *,
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
    ):
        """Fill every network-grid point made available by a source frame."""
        if self.next_sample_t_us is None:
            self.start_at(t_us)
        appended = 0
        while self.next_sample_t_us <= t_us:
            target = self.next_sample_t_us
            self.buffer.add_data_interpolated(
                int(last_t_us),
                int(t_us),
                last_gyr,
                gyr,
                last_displacement,
                displacement,
                last_phase,
                phase,
                last_acc,
                acc,
                int(target),
            )
            self.next_sample_t_us = target + self.sample_period_us
            appended += 1
        return appended

    def input_bounds(self, input_end_us: int):
        """Return inclusive timestamps for a causal window ending at ``k``."""
        input_end_us = int(input_end_us)
        input_begin_us = (
            input_end_us - (self.window_size - 1) * self.sample_period_us
        )
        return input_begin_us, input_end_us

    def get_window(self, input_end_us: int):
        """Extract a complete synchronized C5 window ending at ``input_end_us``."""
        begin_us, end_us = self.input_bounds(input_end_us)
        gyr, displacement, phase, acc, timestamps_us = (
            self.buffer.get_data_from_to(begin_us, end_us)
        )
        if len(timestamps_us) != self.window_size:
            raise RuntimeError(
                f"network window has {len(timestamps_us)} samples; "
                f"expected {self.window_size}"
            )
        return NetworkWindow(
            timestamps_s=np.asarray(timestamps_us, dtype=float) * 1e-6,
            gyr_body=gyr,
            acc_body=acc,
            wing_displacement=displacement,
            wing_phase=phase,
        )

    def discard_before(self, timestamp_us: int):
        self.buffer.throw_data_before(int(timestamp_us))
