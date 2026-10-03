"""Fixed paper C5 network for next-step body displacement prediction."""

import torch
import torch.nn as nn


RAW_INPUT_DIM = 10
OUTPUT_DIM = 3
IMU_ENCODER_DIM = 32
DISPLACEMENT_ENCODER_DIM = 8
PHASE_ENCODER_DIM = 8
FUSION_DIM = 64
TCN_CHANNELS = (64, 64, 64, 64, 128, 128)
TCN_KERNEL_SIZE = 2
GRU_HIDDEN_DIM = 128
GRU_NUM_LAYERS = 2
DECODER_HIDDEN_DIM = 128


class ChannelLayerNorm(nn.Module):
    """Apply layer normalization across channels at each time step."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class PointwiseEncoder(nn.Module):
    """Encode each modality independently at every time step."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("encoder input/output dimensions must be positive")
        self.projection = nn.Conv1d(input_dim, output_dim, kernel_size=1)
        self.norm1 = ChannelLayerNorm(output_dim)
        self.act1 = nn.GELU()
        self.refinement = nn.Conv1d(output_dim, output_dim, kernel_size=1)
        self.norm2 = ChannelLayerNorm(output_dim)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.act1(self.norm1(self.projection(x)))
        refined = self.norm2(self.refinement(hidden))
        return self.act2(hidden + refined)


class Chomp1d(nn.Module):
    """Remove right-side padding introduced by causal convolution.

    Args:
        chomp_size: Number of padded time steps to remove.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size]


class TemporalBlock(nn.Module):
    """Two-layer causal TCN block with a residual connection.

    Args:
        n_inputs: Input channels.
        n_outputs: Output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        dilation: Dilation factor.
        padding:     padding = (kernel_size - 1) * dilation
        dropout: Dropout probability.
    """

    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                 stride: int, dilation: int, padding: int,
                 dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride=stride,
            padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size, stride=stride,
            padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.act_out = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [B, n_inputs, T] → [B, n_outputs, T]"""
        out = self.drop1(self.act1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.act2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.act_out(out + res)


class TemporalConvNet(nn.Module):
    """Stack temporal blocks with exponentially increasing dilation.

    Args:
        num_inputs: Input channels.
        num_hidden_channels: Output channels for the six fixed blocks.
        dropout: Dropout probability.
    """

    def __init__(self, num_inputs: int, num_hidden_channels,
                 dropout: float = 0.1):
        super().__init__()
        num_levels = len(num_hidden_channels)
        assert num_levels > 0, "channels must not be empty"

        layers = []
        in_ch = num_inputs
        for i in range(num_levels):
            dilation = 2 ** i
            padding = (TCN_KERNEL_SIZE - 1) * dilation
            out_ch = num_hidden_channels[i]
            layers.append(TemporalBlock(
                in_ch, out_ch, TCN_KERNEL_SIZE, stride=1, dilation=dilation,
                padding=padding, dropout=dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [B, 10, T] → [B, last_channel, T]"""
        return self.network(x)


class DisplacementDecoder(nn.Module):
    """Decode the next-step body-frame displacement.

    Args:
        input_dim: Decoder input dimension.
        hidden_dim: Hidden dimension.
        output_dim: Output dimension (3 for delta x, y, z).
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 output_dim: int = 3):
        super().__init__()
        self.dp_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """Args: x [B, input_dim] → {'delta_p': [B,3]}."""
        return {"delta_p": self.dp_head(x)}


class LepidNet(nn.Module):
    """Fixed C5 topology: modality encoders, fusion, TCN, GRU, and decoder."""

    def __init__(self, config: dict | None = None):
        super().__init__()
        config = config or {}
        dropout = float(config.get("dropout", 0.1))
        legacy_gru = config.get("gru", {})
        gru_dropout = float(
            config.get("gru_dropout", legacy_gru.get("dropout", 0.1))
        )
        if not 0.0 <= dropout < 1.0 or not 0.0 <= gru_dropout < 1.0:
            raise ValueError("dropout and gru_dropout must be in [0, 1)")

        self.input_dim = RAW_INPUT_DIM
        self.output_dim = OUTPUT_DIM
        self.phase_encoding = "sin_cos"
        self.use_encoders = True
        self.use_fusion_encoder = True
        self.use_gru = True
        self.gru_bidirectional = False
        self.imu_encoder = PointwiseEncoder(6, IMU_ENCODER_DIM)
        self.displacement_encoder = PointwiseEncoder(2, DISPLACEMENT_ENCODER_DIM)
        self.phase_encoder = PointwiseEncoder(4, PHASE_ENCODER_DIM)
        self.feature_fusion = PointwiseEncoder(
            IMU_ENCODER_DIM + DISPLACEMENT_ENCODER_DIM + PHASE_ENCODER_DIM,
            FUSION_DIM,
        )

        self.tcn = TemporalConvNet(
            num_inputs=FUSION_DIM,
            num_hidden_channels=TCN_CHANNELS,
            dropout=dropout,
        )
        self.gru = nn.GRU(
            input_size=TCN_CHANNELS[-1],
            hidden_size=GRU_HIDDEN_DIM,
            num_layers=GRU_NUM_LAYERS,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=False,
        )

        self.decoder = DisplacementDecoder(
            input_dim=GRU_HIDDEN_DIM,
            hidden_dim=DECODER_HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
        )
        self.model_config = {
            "architecture": "C5",
            "input_dim": RAW_INPUT_DIM,
            "phase_encoding": "sin_cos",
            "encoder_dims": {
                "imu": IMU_ENCODER_DIM,
                "flapping_angle": DISPLACEMENT_ENCODER_DIM,
                "phase": PHASE_ENCODER_DIM,
                "fusion": FUSION_DIM,
            },
            "tcn_channels": list(TCN_CHANNELS),
            "tcn_kernel_size": TCN_KERNEL_SIZE,
            "gru_hidden_dim": GRU_HIDDEN_DIM,
            "gru_num_layers": GRU_NUM_LAYERS,
            "gru_bidirectional": False,
            "decoder_hidden_dim": DECODER_HIDDEN_DIM,
            "output_dim": OUTPUT_DIM,
            "dropout": dropout,
            "gru_dropout": gru_dropout,
        }

    def _encode_phase(self, phase: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                torch.sin(phase[:, 0]),
                torch.cos(phase[:, 0]),
                torch.sin(phase[:, 1]),
                torch.cos(phase[:, 1]),
            ],
            dim=1,
        )

    def _encode_input(self, x: torch.Tensor) -> torch.Tensor:
        imu = x[:, 0:6, :]
        displacement = x[:, 6:8, :]
        phase = self._encode_phase(x[:, 8:10, :])
        encoded_modalities = [
            self.imu_encoder(imu),
            self.displacement_encoder(displacement),
            self.phase_encoder(phase),
        ]
        encoded = torch.cat(encoded_modalities, dim=1)
        return self.feature_fusion(encoded)

    def forward(self, x: torch.Tensor) -> dict:
        """Args: x [B, 10, T] → {'delta_p': [B,3]}."""
        if x.ndim != 3 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"model input must have shape [B, {self.input_dim}, T], "
                f"got {tuple(x.shape)}"
            )
        feat = self.tcn(self._encode_input(x))  # [B, C, T]
        _, hidden = self.gru(feat.transpose(1, 2))
        summary = hidden[-1]
        return self.decoder(summary)

    def get_num_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
