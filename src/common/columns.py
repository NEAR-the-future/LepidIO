"""Canonical CSV and network feature schemas."""

CSV_COLUMNS = [
    "timestamp_us",
    "gyr_x", "gyr_y", "gyr_z",
    "acc_x", "acc_y", "acc_z",
    "flap_displacement_left_actual_rad",
    "flap_displacement_right_actual_rad",
    "flap_phase_left_actual_rad",
    "flap_phase_right_actual_rad",
    "gt_px", "gt_py", "gt_pz",
    "gt_qx", "gt_qy", "gt_qz", "gt_qw",
]

# Network input columns in their fixed channel order.
GYR_COLS = ["gyr_x", "gyr_y", "gyr_z"]
ACC_COLS = ["acc_x", "acc_y", "acc_z"]
WING_DISPLACEMENT_COLS = [
    "flap_displacement_left_actual_rad",
    "flap_displacement_right_actual_rad",
]
WING_PHASE_COLS = [
    "flap_phase_left_actual_rad",
    "flap_phase_right_actual_rad",
]
GT_P_COLS = ["gt_px", "gt_py", "gt_pz"]
GT_Q_COLS = ["gt_qx", "gt_qy", "gt_qz", "gt_qw"]
MAG_COLS = ["mag_x", "mag_y", "mag_z"]

# Training and online inference share this exact channel order.
NETWORK_INPUT_COLUMNS = (
    GYR_COLS + ACC_COLS + WING_DISPLACEMENT_COLS + WING_PHASE_COLS
)
FEATURE_SCHEMA = "imu6_actual_wing_displacement2_actual_phase2_v1"
MODEL_ARCHITECTURE_SCHEMA = "full_multimodal_fusion_tcn_gru_displacement_v2"
TARGET_SCHEMA = "next_step_body_delta_position_v1"
NET_INPUT_DIM = 10
