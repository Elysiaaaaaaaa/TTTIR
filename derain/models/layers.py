# ============================================================
# TTTIR
# TTTIR: Unlocking Instance-Specific State Evolution via
# Test-Time Training for Image Restoration
#
# Paper-aligned module hierarchy:
#
#   SESGroup
#       └── SES: State Evolution Stage
#             ├── PSSG: Progressive State Sequence Generation
#             │     ├── PSSC: Progressive Spatial State Constructor
#             │     └── PFSC: Progressive Frequency State Constructor
#             ├── STE: State Transition Evolution
#             │     └── RestorationOrientedTTT: transition operator Phi
#             └── CAB: Channel Attention Block
#
# Important:
#   The paper defines target states as {T_0, ..., T_L}.
#   Therefore:
#       num_states = L + 1
#
#   For the default paper setting L = 2:
#       num_states = 3
# ============================================================

import math
import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath
from pytorch_wavelets import DWTForward


# ============================================================
# Initialization utilities
# ============================================================

def _no_grad_trunc_normal_(
    tensor: torch.Tensor,
    mean: float,
    std: float,
    a: float,
    b: float,
) -> torch.Tensor:
    """Initialize a tensor with values drawn from a truncated normal."""

    def norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn(
            "mean is more than 2 std from [a, b] in "
            "nn.init.trunc_normal_. The distribution of values "
            "may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)

        return tensor


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ============================================================
# Common layers
# ============================================================

class BasicConv(nn.Module):
    """Basic convolution block used by the restoration backbone."""

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        stride: int,
        bias: bool = True,
        norm: bool = False,
        relu: bool = True,
        transpose: bool = False,
    ):
        super().__init__()

        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = []

        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                )
            )

        if norm:
            layers.append(nn.BatchNorm2d(out_channel))

        if relu:
            layers.append(nn.GELU())

        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class ChannelAttention(nn.Module):
    """Channel attention used at the end of an SES."""

    def __init__(self, num_feat: int, squeeze_factor: int = 16):
        super().__init__()

        hidden_dim = max(num_feat // squeeze_factor, 1)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, hidden_dim, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.attention(x)
        return x * attention


class CAB(nn.Module):
    """
    Channel Attention Block.

    In the paper framework, CAB corresponds to the channel-attention
    refinement following progressive state evolution inside an SES.
    """

    def __init__(
        self,
        num_feat: int,
        compress_ratio: int = 2,
        squeeze_factor: int = 30,
    ):
        super().__init__()

        hidden_dim = max(num_feat // compress_ratio, 1)

        self.cab = nn.Sequential(
            nn.Conv2d(
                num_feat,
                hidden_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=hidden_dim,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                num_feat,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2d(
                num_feat,
                num_feat,
                kernel_size=3,
                stride=1,
                padding=2,
                groups=num_feat,
                dilation=2,
            ),
            ChannelAttention(
                num_feat=num_feat,
                squeeze_factor=squeeze_factor,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cab(x)


class PreNorm(nn.Module):
    """Layer normalization before a channel-last operation."""

    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class FeedForward(nn.Module):
    """Convolutional feed-forward refinement used between state levels."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()

        hidden_dim = dim * mult

        self.net = nn.Sequential(
            nn.Conv2d(
                dim,
                hidden_dim,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=hidden_dim,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                dim,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: channel-last feature [B, H, W, C].

        Returns:
            Refined feature [B, H, W, C].
        """
        x = x.permute(0, 3, 1, 2)
        x = self.net(x)
        return x.permute(0, 2, 3, 1)


# ============================================================
# PFSC: Progressive Frequency State Constructor
# ============================================================

class FrequencyBandNorm(nn.Module):
    """
    Per-band normalization used by PFSC.

    Each DWT frequency sub-band is independently normalized across
    its spatial dimensions before frequency feature encoding.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)

        return (x - mean) / (std + self.eps)


class PFSC(nn.Module):
    """
    Progressive Frequency State Constructor.

    Paper correspondence:
        B_m = E(Norm(DWT(X)_m))

        W_l^m = Softmax(Conv([B_m]))

        s_freq^l = Conv(
            sum_m W_l^m * B_m
        )

    Processing:
        1. Apply one-level Haar DWT.
        2. Obtain LL, LH, HL and HH frequency sub-bands.
        3. Normalize and encode each sub-band.
        4. Predict level-specific band fusion weights.
        5. Generate a progressive frequency state sequence.

    Args:
        channels:
            Input feature dimension.

        num_outputs:
            Number of generated frequency states. This argument is
            retained for compatibility with the original implementation.

        reduction:
            Reduction factor in the band-weight predictor.

        num_states:
            Explicit number of progressive states. When provided, it
            overrides num_outputs.

    Shape:
        Input:
            X: [B, C, H, W]

        Output:
            S_freq: [B, K, C, H, W]

        where K = num_states = L + 1.
    """

    def __init__(
        self,
        channels: int,
        num_outputs: int = 3,
        reduction: int = 4,
        num_states: Optional[int] = None,
    ):
        super().__init__()

        if num_states is None:
            num_states = num_outputs

        if num_states < 1:
            raise ValueError(
                f"num_states must be positive, but got {num_states}."
            )

        self.channels = channels
        self.num_states = num_states

        # One-level Haar discrete wavelet transform.
        self.dwt = DWTForward(
            J=1,
            wave="haar",
            mode="zero",
        )

        hidden_dim = max(channels // reduction, 8)

        # Shared encoder E(.) in Eq. (6).
        self.band_encoder = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )

        # Predict K groups of four frequency-band weights.
        self.band_weight_predictor = nn.Sequential(
            nn.Conv2d(
                channels * 4,
                hidden_dim,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                num_states * 4,
                kernel_size=1,
            ),
        )

        self.band_norm = FrequencyBandNorm()

        # Restore the DWT feature resolution from H/2 x W/2 to H x W.
        self.upsample = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
            ),
        )

        # Final convolution in Eq. (8).
        self.state_refinement = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
            ),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate progressive frequency states.

        Args:
            x: Input feature [B, C, H, W].

        Returns:
            frequency_states:
                Progressive frequency states [B, K, C, H, W].
        """
        batch_size, channels, height, width = x.shape

        # ----------------------------------------------------
        # Step 1: DWT decomposition
        # ----------------------------------------------------
        ll, high_frequency = self.dwt(x)

        # pytorch_wavelets ordering:
        # high_frequency[0][:, :, 0] -> LH
        # high_frequency[0][:, :, 1] -> HL
        # high_frequency[0][:, :, 2] -> HH
        lh = high_frequency[0][:, :, 0]
        hl = high_frequency[0][:, :, 1]
        hh = high_frequency[0][:, :, 2]

        # ----------------------------------------------------
        # Step 2: independent band normalization
        # ----------------------------------------------------
        ll = self.band_norm(ll)
        lh = self.band_norm(lh)
        hl = self.band_norm(hl)
        hh = self.band_norm(hh)

        # ----------------------------------------------------
        # Step 3: shared frequency-band encoder
        #
        # B_m = E(Norm(DWT(X)_m))
        # ----------------------------------------------------
        ll_encoded = self.band_encoder(ll)
        lh_encoded = self.band_encoder(lh)
        hl_encoded = self.band_encoder(hl)
        hh_encoded = self.band_encoder(hh)

        encoded_bands = torch.stack(
            [
                ll_encoded,
                lh_encoded,
                hl_encoded,
                hh_encoded,
            ],
            dim=1,
        )
        # encoded_bands: [B, 4, C, H/2, W/2]

        # ----------------------------------------------------
        # Step 4: level-specific adaptive band weighting
        #
        # W_l^m = Softmax(Conv([B_m]))
        # ----------------------------------------------------
        concatenated_bands = torch.cat(
            [
                ll_encoded,
                lh_encoded,
                hl_encoded,
                hh_encoded,
            ],
            dim=1,
        )

        band_logits = self.band_weight_predictor(concatenated_bands)

        sub_height = band_logits.shape[-2]
        sub_width = band_logits.shape[-1]

        band_logits = band_logits.reshape(
            batch_size,
            self.num_states,
            4,
            sub_height,
            sub_width,
        )

        band_weights = F.softmax(band_logits, dim=2)
        # band_weights: [B, K, 4, H/2, W/2]

        # ----------------------------------------------------
        # Step 5: generate level-wise frequency states
        #
        # s_freq^l = Conv(sum_m W_l^m * B_m)
        # ----------------------------------------------------
        frequency_states = []

        for level in range(self.num_states):
            level_weights = band_weights[:, level]
            # [B, 4, H/2, W/2]

            level_weights = level_weights.unsqueeze(2)
            # [B, 4, 1, H/2, W/2]

            state = (encoded_bands * level_weights).sum(dim=1)
            # [B, C, H/2, W/2]

            state = self.state_refinement(state)
            state = self.upsample(state)

            # Handle odd input resolutions introduced by DWT/upsampling.
            if state.shape[-2:] != (height, width):
                state = F.interpolate(
                    state,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )

            frequency_states.append(state)

        frequency_states = torch.stack(
            frequency_states,
            dim=1,
        )
        # [B, K, C, H, W]

        return frequency_states


# ============================================================
# PSSC: Progressive Spatial State Constructor
# ============================================================

class PSSC(nn.Module):
    """
    Progressive Spatial State Constructor.

    Paper correspondence:
        [C_0, G_0, ..., G_L] = Conv_1x1(X)

        C_tilde_l = DWConv_l(C_l)

        C_{l+1} = gamma_l * C_l + C_tilde_l

        s_l = C_{l+1} * G_l

    The generated sequence is reversed so that earlier states contain
    broader structural information and later states focus on local
    texture and edge refinement.

    Args:
        dim:
            Input feature dimension.

        focal_level:
            Original argument retained for compatibility. In this code,
            it denotes the number of states rather than the maximum
            state index L.

        num_states:
            Explicit number of progressive states. When provided, it
            overrides focal_level.

    Shape:
        Input:
            X: [B, C, H, W]

        Output:
            S_space: [B, K, C, H, W]

        where K = num_states = L + 1.
    """

    def __init__(
        self,
        dim: int,
        proj_drop: float = 0.0,
        focal_level: int = 3,
        focal_window: int = 7,
        focal_factor: int = 2,
        use_postln: bool = False,
        num_states: Optional[int] = None,
    ):
        super().__init__()

        if num_states is None:
            num_states = focal_level

        if num_states < 1:
            raise ValueError(
                f"num_states must be positive, but got {num_states}."
            )

        self.dim = dim
        self.num_states = num_states
        self.focal_window = focal_window
        self.focal_factor = focal_factor
        self.use_postln = use_postln

        # Conv_1x1 generates:
        #   initial context C_0: C channels
        #   gates G_0 ... G_{K-1}: K channels
        self.context_gate_projection = nn.Conv2d(
            dim,
            dim + num_states,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=True,
        )

        self.spatial_constructors = nn.ModuleList()
        self.proj_drop = nn.Dropout(proj_drop)

        if use_postln:
            self.post_norm = nn.LayerNorm(dim)

        # Learnable gamma_l in Eq. (3).
        self.context_scales = nn.Parameter(
            torch.zeros(
                num_states,
                1,
                dim,
                1,
                1,
            )
        )

        for level in range(num_states):
            kernel_size = (
                self.focal_factor * level
                + self.focal_window
            )

            self.spatial_constructors.append(
                nn.Sequential(
                    nn.Conv2d(
                        dim,
                        dim,
                        kernel_size=kernel_size,
                        stride=1,
                        groups=dim,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.GELU(),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate progressive spatial states.

        Args:
            x: Input feature [B, C, H, W].

        Returns:
            spatial_states:
                Progressive spatial states [B, K, C, H, W].
        """
        _, channels, _, _ = x.shape

        # ----------------------------------------------------
        # [C_0, G_0, ..., G_L] = Conv_1x1(X)
        # ----------------------------------------------------
        projected = self.context_gate_projection(x)

        context, gates = torch.split(
            projected,
            [channels, self.num_states],
            dim=1,
        )

        spatial_states = []

        for level in range(self.num_states):
            # C_tilde_l = DWConv_l(C_l)
            enhanced_context = self.spatial_constructors[level](
                context
            )

            # C_{l+1} = gamma_l * C_l + C_tilde_l
            context = (
                context * self.context_scales[level]
                + self.proj_drop(enhanced_context)
            )

            # s_l = C_{l+1} * G_l
            level_gate = gates[:, level:level + 1]
            spatial_state = context * level_gate

            spatial_states.append(spatial_state)

        # Reverse the sequence to follow the coarse-to-fine trajectory.
        spatial_states.reverse()

        spatial_states = torch.stack(
            spatial_states,
            dim=1,
        )
        # [B, K, C, H, W]

        return spatial_states


# ============================================================
# PSSG: Progressive State Sequence Generation
# ============================================================

class PSSG(nn.Module):
    """
    Progressive State Sequence Generation.

    PSSG combines the spatial state sequence generated by PSSC and
    the frequency state sequence generated by PFSC:

        T_l = alpha_l * s_space^l
              + (1 - alpha_l) * s_freq^l

    The paper uses alpha = 0.5 in its default implementation.

    Args:
        pssc:
            Progressive Spatial State Constructor.

        pfsc:
            Progressive Frequency State Constructor.

        alpha:
            Spatial-frequency fusion coefficient. It can be:
                - a scalar shared by all levels;
                - a sequence containing one coefficient per state.

    Returns:
        target_states:
            T = {T_0, ..., T_L}.

        spatial_states:
            S_space.

        frequency_states:
            S_freq.
    """

    def __init__(
        self,
        pssc: PSSC,
        pfsc: PFSC,
        alpha: Union[float, Sequence[float]] = 0.5,
    ):
        super().__init__()

        if pssc.num_states != pfsc.num_states:
            raise ValueError(
                "PSSC and PFSC must generate the same number of states, "
                f"but got {pssc.num_states} and {pfsc.num_states}."
            )

        self.pssc = pssc
        self.pfsc = pfsc
        self.num_states = pssc.num_states

        alpha_tensor = torch.as_tensor(
            alpha,
            dtype=torch.float32,
        )

        if alpha_tensor.ndim == 0:
            alpha_tensor = alpha_tensor.repeat(self.num_states)
        else:
            alpha_tensor = alpha_tensor.flatten()

        if alpha_tensor.numel() != self.num_states:
            raise ValueError(
                "alpha must be a scalar or contain one value per state. "
                f"Expected {self.num_states} values, but got "
                f"{alpha_tensor.numel()}."
            )

        if torch.any(alpha_tensor < 0) or torch.any(alpha_tensor > 1):
            raise ValueError(
                "Every alpha value must be in the interval [0, 1]."
            )

        # Shape: [1, K, 1, 1, 1]
        self.register_buffer(
            "alpha",
            alpha_tensor.reshape(
                1,
                self.num_states,
                1,
                1,
                1,
            ),
            persistent=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input feature [B, C, H, W].

        Returns:
            target_states:
                [B, K, C, H, W]

            spatial_states:
                [B, K, C, H, W]

            frequency_states:
                [B, K, C, H, W]
        """
        spatial_states = self.pssc(x)
        frequency_states = self.pfsc(x)

        target_states = (
            self.alpha * spatial_states
            + (1.0 - self.alpha) * frequency_states
        )

        return (
            target_states,
            spatial_states,
            frequency_states,
        )


# ============================================================
# Restoration-oriented TTT transition operator Phi
# ============================================================

class RestorationOrientedTTT(nn.Module):
    """
    Restoration-oriented Test-Time Training transition operator.

    This module implements the lightweight transition function Phi
    used by STE.

    For the current state Z_l and next target state T_{l+1}:

        Q_l, K_l = MLP_in(Z_l)
        V_l      = MLP_tar(T_{l+1})

        W_{l+1} = W_l - eta * dL/dW

        Q_hat_l = Phi_{W_{l+1}}(Q_l)

    The transition function is implemented as an input-conditioned
    3x3 depth-wise convolution whose fast weights are updated by the
    restoration-oriented inner-loop objective.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        inner_lr: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}."
            )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.inner_lr = inner_lr

        # Generate Q_l and K_l from the current state Z_l.
        self.current_state_projection = nn.Linear(
            dim,
            self.head_dim * 2,
            bias=qkv_bias,
        )

        # Generate V_l from the next target state T_{l+1}.
        self.target_state_projection = nn.Linear(
            dim,
            self.head_dim,
            bias=qkv_bias,
        )

        # Initialization W_0 of the lightweight transition operator Phi.
        self.initial_transition_weights = nn.Parameter(
            torch.zeros(
                self.head_dim,
                1,
                3,
                3,
            )
        )
        trunc_normal_(
            self.initial_transition_weights,
            std=0.02,
        )

        # Project the evolved head feature back to dimension C.
        self.output_projection = nn.Linear(
            self.head_dim,
            dim,
        )

        # Equivalent dimension of a single-channel 3x3 kernel.
        equivalent_head_dim = 9
        self.scale = equivalent_head_dim ** -0.5

    def update_transition_weights(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        weights: torch.Tensor,
        lr: Optional[float] = None,
        implementation: str = "prod",
    ) -> torch.Tensor:
        """
        Perform the restoration-oriented inner-loop update.

        Args:
            key:
                K_l with shape [B, D, H, W].

            value:
                V_l with shape [B, D, H, W].

            weights:
                Current transition weights W_l.

                Initial weights:
                    [D, 1, 3, 3]

                Instance-specific fast weights:
                    [B * D, 1, 3, 3]

            lr:
                Inner-loop learning rate eta.

            implementation:
                "prod" uses explicit shifted products.
                "conv" uses grouped convolution.

        Returns:
            Updated transition weights W_{l+1}.
        """
        if lr is None:
            lr = self.inner_lr

        batch_size, channels, height, width = key.shape

        # Gradient signal induced by the dot-product inner-loop loss.
        error = (
            -value
            / float(value.shape[2] * value.shape[3])
            * self.scale
        )

        if implementation == "conv":
            gradient = F.conv2d(
                key.reshape(
                    1,
                    batch_size * channels,
                    height,
                    width,
                ),
                error.reshape(
                    batch_size * channels,
                    1,
                    height,
                    width,
                ),
                padding=1,
                groups=batch_size * channels,
            )

            gradient = gradient.transpose(0, 1)

        elif implementation == "prod":
            padded_key = F.pad(
                key,
                pad=(1, 1, 1, 1),
            )

            shifted_products = []

            for delta_y in (-1, 0, 1):
                for delta_x in (-1, 0, 1):
                    y_start = 1 + delta_y
                    x_start = 1 + delta_x

                    shifted_key = padded_key[
                        :,
                        :,
                        y_start:y_start + height,
                        x_start:x_start + width,
                    ]

                    dot_product = (
                        shifted_key * error
                    ).sum(dim=(-2, -1))

                    shifted_products.append(dot_product)

            gradient = torch.stack(
                shifted_products,
                dim=-1,
            )

            gradient = gradient.reshape(
                batch_size * channels,
                1,
                3,
                3,
            )

        else:
            raise NotImplementedError(
                f"Unsupported implementation: {implementation}"
            )

        # Stabilize the instance-specific inner-loop gradient.
        gradient = gradient / (
            gradient.norm(
                dim=(-2, -1),
                keepdim=True,
            )
            + 1.0
        )

        # The initial W_0 is shared across images and therefore has D
        # kernels. It is expanded to B * D instance-specific kernels.
        if weights.shape[0] == self.head_dim:
            weights = weights.repeat(
                batch_size,
                1,
                1,
                1,
            )

        updated_weights = weights - lr * gradient

        return updated_weights

    def forward(
        self,
        current_state: torch.Tensor,
        next_target_state: Optional[torch.Tensor] = None,
        fast_weights: Optional[torch.Tensor] = None,
        target_state: Optional[torch.Tensor] = None,
        w1=None,
        w2=None,
        w3: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evolve Z_l toward T_{l+1}.

        Args:
            current_state:
                Current restoration state Z_l, [B, H, W, C].

            next_target_state:
                Next progressive target state T_{l+1},
                [B, H, W, C].

            fast_weights:
                Previously updated transition weights W_l.

            target_state, w3:
                Compatibility arguments corresponding to the original
                implementation. New code should use next_target_state
                and fast_weights.

        Returns:
            evolved_feature:
                Q_hat_l, [B, H, W, C].

            updated_fast_weights:
                W_{l+1}, [B * D, 1, 3, 3].
        """
        # Compatibility with the original forward signature.
        if next_target_state is None:
            next_target_state = target_state

        if fast_weights is None:
            fast_weights = w3

        if next_target_state is None:
            raise ValueError(
                "next_target_state must be provided for "
                "restoration-oriented state transition."
            )

        # The target branch only supplies inner-loop supervision.
        next_target_state = next_target_state.detach()

        if current_state.ndim != 4:
            raise ValueError(
                "current_state must have shape [B, H, W, C], "
                f"but got {tuple(current_state.shape)}."
            )

        batch_size, height, width, channels = current_state.shape
        num_tokens = height * width

        current_tokens = current_state.reshape(
            batch_size,
            num_tokens,
            channels,
        )

        target_tokens = next_target_state.reshape(
            batch_size,
            num_tokens,
            channels,
        )

        # ----------------------------------------------------
        # Q_l, K_l = MLP_in(Z_l)
        # V_l      = MLP_tar(T_{l+1})
        # ----------------------------------------------------
        query, key = torch.split(
            self.current_state_projection(current_tokens),
            [self.head_dim, self.head_dim],
            dim=-1,
        )

        value = self.target_state_projection(target_tokens)

        query = query.reshape(
            batch_size,
            height,
            width,
            self.head_dim,
        ).permute(0, 3, 1, 2)

        key = key.reshape(
            batch_size,
            height,
            width,
            self.head_dim,
        ).permute(0, 3, 1, 2)

        value = value.reshape(
            batch_size,
            height,
            width,
            self.head_dim,
        ).permute(0, 3, 1, 2)

        # ----------------------------------------------------
        # W_{l+1} = W_l - eta * dL/dW
        # ----------------------------------------------------
        if fast_weights is None:
            transition_weights = self.initial_transition_weights
        else:
            transition_weights = fast_weights

        updated_fast_weights = self.update_transition_weights(
            key=key,
            value=value,
            weights=transition_weights,
            implementation="prod",
        )

        # ----------------------------------------------------
        # Q_hat_l = Phi_{W_{l+1}}(Q_l)
        # ----------------------------------------------------
        evolved_query = F.conv2d(
            query.reshape(
                1,
                batch_size * self.head_dim,
                height,
                width,
            ),
            updated_fast_weights,
            padding=1,
            groups=batch_size * self.head_dim,
        )

        evolved_query = evolved_query.reshape(
            batch_size,
            self.head_dim,
            num_tokens,
        ).transpose(1, 2)

        evolved_feature = self.output_projection(evolved_query)

        evolved_feature = evolved_feature.reshape(
            batch_size,
            height,
            width,
            channels,
        )

        return evolved_feature, updated_fast_weights


# ============================================================
# STE: State Transition Evolution
# ============================================================

class STE(nn.Module):
    """
    State Transition Evolution.

    STE progressively evolves restoration states:

        Z_0 -> Z_1 -> ... -> Z_L

    At transition level l:
        1. The current state Z_l generates Q_l and K_l.
        2. The next target state T_{l+1} generates V_l.
        3. The transition operator Phi is updated by the inner loop.
        4. The updated operator is applied to Q_l.
        5. Residual refinement generates Z_{l+1}.

    The first target state T_0 is used as the initial restoration
    state, matching the behavior of the original implementation.
    """

    def __init__(
        self,
        d_model: int,
        num_states: int,
        inner_lr: float = 1.0,
    ):
        super().__init__()

        if num_states < 2:
            raise ValueError(
                "STE requires at least two states, but got "
                f"num_states={num_states}."
            )

        if d_model % 8 != 0:
            raise ValueError(
                "The current implementation uses head_dim=8, so "
                f"d_model={d_model} must be divisible by 8."
            )

        self.d_model = d_model
        self.num_states = num_states

        self.transition_operator = RestorationOrientedTTT(
            dim=d_model,
            num_heads=d_model // 8,
            inner_lr=inner_lr,
        )

        # Preserve the parameterization of the original implementation:
        # one refinement block is allocated for every state, while
        # blocks indexed from 1 to K-1 are used after transitions.
        self.state_refinement_blocks = nn.ModuleList(
            [
                PreNorm(
                    d_model,
                    FeedForward(dim=d_model),
                )
                for _ in range(num_states)
            ]
        )

        self.transition_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        target_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            target_states:
                Target sequence T with shape [B, K, C, H, W].

        Returns:
            evolved_state:
                Final state Z_L with shape [B, C, H, W].
        """
        if target_states.ndim != 5:
            raise ValueError(
                "target_states must have shape [B, K, C, H, W], "
                f"but got {tuple(target_states.shape)}."
            )

        _, num_states, _, _, _ = target_states.shape

        if num_states != self.num_states:
            raise ValueError(
                f"STE expects {self.num_states} states, "
                f"but received {num_states}."
            )

        # Convert to channel-last format for LayerNorm and linear layers.
        target_states = target_states.permute(
            0,
            1,
            3,
            4,
            2,
        )
        # [B, K, H, W, C]

        # T_0 initializes the restoration trajectory.
        current_state = target_states[:, 0]

        # W_0 is used at the first level and progressively updated.
        fast_weights = None

        for level in range(1, self.num_states):
            next_target_state = target_states[:, level]

            # Restoration-oriented inner-loop adaptation.
            evolved_feature, fast_weights = self.transition_operator(
                current_state=current_state,
                next_target_state=next_target_state,
                fast_weights=fast_weights,
            )

            # R_l = Norm(Q_hat_l) + Z_l
            evolved_feature = self.transition_norm(evolved_feature)
            current_state = current_state + evolved_feature

            # Z_{l+1} = Conv(R_l) + R_l
            current_state = (
                current_state
                + self.state_refinement_blocks[level](
                    current_state
                )
            )

        evolved_state = current_state.permute(
            0,
            3,
            1,
            2,
        )

        return evolved_state


# ============================================================
# SES: State Evolution Stage
# ============================================================

class SES(nn.Module):
    """
    State Evolution Stage.

    Paper-aligned processing order:

        X
        -> LayerNorm
        -> PSSG
             -> PSSC generates S_space
             -> PFSC generates S_freq
             -> fuse into target states T
        -> STE
             -> restoration-oriented inner-loop transition
        -> feature modulation and residual fusion
        -> Channel Attention
        -> output Y_i

    This class corresponds to the original HSEBlock.
    """

    def __init__(
        self,
        hidden_dim: int,
        drop_path: float,
        pssg: PSSG,
        inner_lr: float = 1.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_states = pssg.num_states

        self.pre_norm = nn.LayerNorm(hidden_dim)

        self.pssg = pssg

        self.ste = STE(
            d_model=hidden_dim,
            num_states=self.num_states,
            inner_lr=inner_lr,
        )

        self.drop_path = DropPath(drop_path)

        self.state_skip_scale = nn.Parameter(
            torch.ones(
                1,
                hidden_dim,
                1,
                1,
            )
        )

        self.channel_attention = CAB(hidden_dim)

        self.channel_norm = nn.LayerNorm(hidden_dim)

        self.channel_skip_scale = nn.Parameter(
            torch.ones(
                1,
                hidden_dim,
                1,
                1,
            )
        )

    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_feature:
                SES input [B, C, H, W].

        Returns:
            output_feature:
                SES output [B, C, H, W].
        """
        # ----------------------------------------------------
        # Pre-normalization
        # ----------------------------------------------------
        normalized_feature = self.pre_norm(
            input_feature.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)

        # ----------------------------------------------------
        # PSSG:
        #   S_space, S_freq -> T
        # ----------------------------------------------------
        target_states, _, _ = self.pssg(normalized_feature)

        # ----------------------------------------------------
        # STE:
        #   Z_0 -> ... -> Z_L
        # ----------------------------------------------------
        evolved_state = self.ste(target_states)

        # State-conditioned feature modulation.
        state_feature = normalized_feature * evolved_state

        # Residual fusion after state evolution.
        state_feature = (
            input_feature * self.state_skip_scale
            + self.drop_path(state_feature)
        )

        # Channel-attention refinement.
        normalized_channel_feature = self.channel_norm(
            state_feature.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)

        output_feature = (
            state_feature * self.channel_skip_scale
            + self.channel_attention(normalized_channel_feature)
        )

        return output_feature


# ============================================================
# SESGroup: stacked State Evolution Stages
# ============================================================

class SESGroup(nn.Module):
    """
    Residual group containing multiple State Evolution Stages.

    This class corresponds to the original SEBlock. Each SES follows
    the paper-defined PSSG -> STE -> Channel Attention pipeline.

    Args:
        embed_dim:
            Feature dimension.

        depths:
            Number of cascaded SES modules in this residual group.

        focal_level:
            Compatibility name from the original implementation.
            It denotes the number of generated states K, not the
            maximum state index L.

        num_states:
            Explicit paper-aligned state count K = L + 1.
            When provided, it overrides focal_level.

        alpha:
            Spatial-frequency fusion coefficient in PSSG.

        inner_lr:
            Inner-loop learning rate eta in STE.

    Example:
        The paper uses L=2, so instantiate this module with:

            SESGroup(
                embed_dim=32,
                depths=2,
                num_states=3,
            )
    """

    def __init__(
        self,
        embed_dim: int,
        smooth: bool = False,
        depths: Optional[int] = None,
        drop_path_rate: float = 0.1,
        focal_level: int = 3,
        num_states: Optional[int] = None,
        alpha: Union[float, Sequence[float]] = 0.5,
        inner_lr: float = 1.0,
    ):
        super().__init__()

        if depths is None:
            raise ValueError("depths must be specified.")

        if depths < 1:
            raise ValueError(
                f"depths must be positive, but got {depths}."
            )

        if num_states is None:
            num_states = focal_level

        self.embed_dim = embed_dim
        self.depths = depths
        self.num_states = num_states

        drop_path_rates = [
            value.item()
            for value in torch.linspace(
                0,
                drop_path_rate,
                depths,
            )
        ]

        # ----------------------------------------------------
        # PSSG components shared by the SESs in this group,
        # preserving the behavior of the original implementation.
        # ----------------------------------------------------
        shared_pssc = PSSC(
            dim=embed_dim,
            proj_drop=0.0,
            focal_window=3,
            focal_factor=0,
            use_postln=False,
            num_states=num_states,
        )

        shared_pfsc = PFSC(
            channels=embed_dim,
            num_states=num_states,
        )

        shared_pssg = PSSG(
            pssc=shared_pssc,
            pfsc=shared_pfsc,
            alpha=alpha,
        )

        self.stages = nn.ModuleList()

        for stage_index in range(depths):
            self.stages.append(
                SES(
                    hidden_dim=embed_dim,
                    drop_path=drop_path_rates[stage_index],
                    pssg=shared_pssg,
                    inner_lr=inner_lr,
                )
            )

        self.output_projection = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                Input feature [B, C, H, W].

        Returns:
            Residually refined feature [B, C, H, W].
        """
        residual = x

        for stage in self.stages:
            x = stage(x)

        return self.output_projection(x) + residual


# ============================================================
# Compatibility wrappers for the original class names
# ============================================================

class HSE(nn.Module):
    """
    Compatibility wrapper for the original HSE class.

    New code should explicitly use:
        PSSG + STE
    """

    def __init__(
        self,
        d_model: int,
        focal_level: int,
        space_Foc: nn.Module,
        freq_Foc: nn.Module,
        alpha: float = 0.5,
        inner_lr: float = 1.0,
    ):
        super().__init__()

        self.pssg = PSSG(
            pssc=space_Foc,
            pfsc=freq_Foc,
            alpha=alpha,
        )

        self.ste = STE(
            d_model=d_model,
            num_states=focal_level,
            inner_lr=inner_lr,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_states, _, _ = self.pssg(x)
        return self.ste(target_states)


class HSEBlock(SES):
    """
    Compatibility wrapper for the original HSEBlock class.

    New code should use SES.
    """

    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0.0,
        space_Foc: Optional[PSSC] = None,
        freq_Foc: Optional[PFSC] = None,
        focal_level: Optional[int] = None,
        alpha: float = 0.5,
        inner_lr: float = 1.0,
    ):
        if space_Foc is None or freq_Foc is None:
            raise ValueError(
                "space_Foc and freq_Foc must be provided to the "
                "compatibility HSEBlock."
            )

        if focal_level is None:
            focal_level = space_Foc.num_states

        pssg = PSSG(
            pssc=space_Foc,
            pfsc=freq_Foc,
            alpha=alpha,
        )

        super().__init__(
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            pssg=pssg,
            inner_lr=inner_lr,
        )


class SEBlock(SESGroup):
    """
    Compatibility wrapper for the original SEBlock class.

    New code should use SESGroup.
    """

    def __init__(
        self,
        embed_dim: int,
        smooth: bool = False,
        depths: Optional[int] = None,
        drop_path_rate: float = 0.1,
        focal_level: int = 3,
        num_states: Optional[int] = None,
        alpha: float = 0.5,
        inner_lr: float = 1.0,
    ):
        super().__init__(
            embed_dim=embed_dim,
            smooth=smooth,
            depths=depths,
            drop_path_rate=drop_path_rate,
            focal_level=focal_level,
            num_states=num_states,
            alpha=alpha,
            inner_lr=inner_lr,
        )


# Original-name aliases retained for code-level compatibility.
BandNorm = FrequencyBandNorm
HFSC = PFSC
HSSC = PSSC
TTT = RestorationOrientedTTT