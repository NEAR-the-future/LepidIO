"""Shared utilities for LepidIO dataset paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "Dataset_clean_raw_attitude"
DEFAULT_FIGURE_DIR = REPO_ROOT / "figures" / "dataset_characteristics"

COLORS = {
    "navy": "#005F73",
    "teal": "#0A9396",
    "mint": "#AFD2B7",
    "blue": "#337BAC",
    "cream": "#E6D9A5",
    "orange": "#EE9B00",
    "burnt": "#CA6702",
    "red": "#AE2012",
    "dark": "#202124",
    "gray": "#7B8794",
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def configure_style(axis_label_size: float = 11.0, tick_size: float = 9.5) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.labelsize": axis_label_size,
            "axes.titlesize": 10.0,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": 8.8,
            "axes.linewidth": 0.75,
            "grid.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color="#D8DEE3", alpha=0.65, linewidth=0.55)
    ax.set_axisbelow(True)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16, 1.08, label, transform=ax.transAxes,
        ha="left", va="bottom", fontsize=11.0, fontweight="bold",
        color=COLORS["dark"],
    )


def quaternion_to_euler_xyz_deg(frame: pd.DataFrame) -> np.ndarray:
    x = frame["gt_qx"].to_numpy(dtype=float)
    y = frame["gt_qy"].to_numpy(dtype=float)
    z = frame["gt_qz"].to_numpy(dtype=float)
    w = frame["gt_qw"].to_numpy(dtype=float)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.rad2deg(np.column_stack((roll, pitch, yaw)))


def load_clean_sequence(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    euler = quaternion_to_euler_xyz_deg(frame)
    frame = frame.copy()
    frame["roll_deg"] = euler[:, 0]
    frame["pitch_deg"] = euler[:, 1]
    frame["yaw_deg"] = euler[:, 2]
    frame["time_s"] = frame["timestamp_us"].to_numpy(dtype=float) * 1e-6
    return frame


def load_all_clean(dataset_dir: Path) -> list[tuple[str, pd.DataFrame]]:
    files = sorted(dataset_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")
    return [(path.stem, load_clean_sequence(path)) for path in files]


def balanced_values(
    sequences: list[tuple[str, pd.DataFrame]],
    columns: list[str],
    samples_per_sequence: int = 1200,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pieces: list[pd.DataFrame] = []
    for name, frame in sequences:
        count = min(samples_per_sequence, len(frame))
        indices = (
            np.arange(len(frame)) if count == len(frame)
            else np.sort(rng.choice(len(frame), count, replace=False))
        )
        piece = frame.iloc[indices][columns].copy()
        piece["sequence"] = name
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def periodogram(values: np.ndarray, sample_hz: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values - moving_average(values, max(3, round(sample_hz * 1.0)))
    window = np.hanning(len(values))
    spectrum = np.fft.rfft(values * window)
    power = np.abs(spectrum) ** 2 / max(sample_hz * np.sum(window**2), 1e-12)
    frequency = np.fft.rfftfreq(len(values), d=1.0 / sample_hz)
    return frequency, power


def save_figure(fig: plt.Figure, stem: Path, dpi: int = 400) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [stem.with_suffix(ext) for ext in (".pdf", ".svg", ".png")]
    fig.savefig(outputs[0], bbox_inches="tight", pad_inches=0.035)
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.035)
    fig.savefig(outputs[2], dpi=dpi, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    for output in outputs:
        print(f"Saved: {output}")
    return outputs


def save_panels(
    fig: plt.Figure,
    panels: dict[str, list[plt.Axes] | tuple[plt.Axes, ...] | plt.Axes],
    output_dir: Path,
    prefix: str,
    dpi: int = 400,
    padding_in: float = 0.06,
) -> list[Path]:
    """Export axes from an existing composite figure as separate files.

    A panel may contain multiple axes (for example a main axis and ``twinx``
    axis). Their tight bounding boxes are combined before export.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for panel_name, panel_axes in panels.items():
        axes_list = (
            [panel_axes] if isinstance(panel_axes, plt.Axes)
            else list(panel_axes)
        )
        selected = set(axes_list)
        visibility = {axis: axis.get_visible() for axis in fig.axes}
        panel_label_visibility: dict[object, bool] = {}
        for axis in fig.axes:
            axis.set_visible(axis in selected)
        # Composite figures keep their (a), (b), ... labels. Standalone panel
        # exports omit them so the panels can be rearranged without duplicate
        # numbering.
        for axis in axes_list:
            for artist in axis.texts:
                label = artist.get_text().strip()
                if len(label) == 3 and label[0] == "(" and label[2] == ")" \
                        and label[1].isalpha():
                    panel_label_visibility[artist] = artist.get_visible()
                    artist.set_visible(False)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        pixel_bbox = Bbox.union(
            [axis.get_tightbbox(renderer) for axis in axes_list]
        )
        bbox = pixel_bbox.transformed(fig.dpi_scale_trans.inverted())
        bbox = Bbox.from_extents(
            bbox.x0 - padding_in,
            bbox.y0 - padding_in,
            bbox.x1 + padding_in,
            bbox.y1 + padding_in,
        )
        stem = output_dir / f"{prefix}_{panel_name}"
        panel_outputs = [stem.with_suffix(ext) for ext in (".pdf", ".svg", ".png")]
        fig.savefig(panel_outputs[0], bbox_inches=bbox, pad_inches=0)
        fig.savefig(panel_outputs[1], bbox_inches=bbox, pad_inches=0)
        fig.savefig(panel_outputs[2], dpi=dpi, bbox_inches=bbox, pad_inches=0)
        for axis, was_visible in visibility.items():
            axis.set_visible(was_visible)
        for artist, was_visible in panel_label_visibility.items():
            artist.set_visible(was_visible)
        outputs.extend(panel_outputs)
        print(f"Saved panel: {panel_outputs[2]}")
    fig.canvas.draw()
    return outputs
