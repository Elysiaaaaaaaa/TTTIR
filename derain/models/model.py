import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .layers import BasicConv, SESGroup
except ImportError:
    from layers import BasicConv, SESGroup


# ============================================================
# Shallow Context Module
# ============================================================

class ShallowContextModule(nn.Module):
    """
    Shallow Context Module.

    Extracts shallow features from downsampled RGB inputs and projects
    them to the feature dimension required by each encoder level.

    Args:
        out_channels:
            Output feature dimension.

    Input:
        x: [B, 3, H, W]

    Output:
        feature: [B, out_channels, H, W]
    """

    def __init__(self, out_channels: int):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            BasicConv(
                3,
                out_channels // 4,
                kernel_size=3,
                stride=1,
                relu=True,
                bias=False,
                norm=True,
            ),
            BasicConv(
                out_channels // 4,
                out_channels // 2,
                kernel_size=1,
                stride=1,
                relu=True,
                bias=False,
                norm=True,
            ),
            BasicConv(
                out_channels // 2,
                out_channels // 2,
                kernel_size=3,
                stride=1,
                relu=True,
                bias=False,
                norm=True,
            ),
            BasicConv(
                out_channels // 2,
                out_channels,
                kernel_size=1,
                stride=1,
                relu=False,
            ),
            nn.InstanceNorm2d(
                out_channels,
                affine=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)


# ============================================================
# Feature Aggregation Module
# ============================================================

class FeatureAggregationModule(nn.Module):
    """
    Feature Aggregation Module.

    Aggregates the encoder feature with the shallow context feature
    extracted from the corresponding image scale.

    Args:
        channels:
            Feature dimension of both input branches.

    Inputs:
        encoder_feature: [B, C, H, W]
        context_feature: [B, C, H, W]

    Output:
        aggregated_feature: [B, C, H, W]
    """

    def __init__(self, channels: int):
        super().__init__()

        self.feature_fusion = BasicConv(
            channels * 2,
            channels,
            kernel_size=3,
            stride=1,
            relu=False,
        )

    def forward(
        self,
        encoder_feature: torch.Tensor,
        context_feature: torch.Tensor,
    ) -> torch.Tensor:
        aggregated_feature = torch.cat(
            [encoder_feature, context_feature],
            dim=1,
        )

        return self.feature_fusion(aggregated_feature)


# ============================================================
# TTTIR Encoder
# ============================================================

class TTTIREncoder(nn.Module):
    """
    Encoder of TTTIR.

    The encoder progressively extracts multi-scale restoration
    representations. Each encoder level contains an SESGroup composed
    of multiple State Evolution Stages.

    Args:
        base_channels:
            Feature dimension of the first encoder level.

        num_blocks:
            Number of SES modules at the three encoder levels.

        num_states:
            Number of progressive target states generated in each SES.
            The paper setting L=2 corresponds to num_states=3.

        alpha:
            Spatial-frequency state fusion coefficient in PSSG.

        inner_lr:
            Inner-loop learning rate used in STE.

        smooth:
            Compatibility argument retained from the original code.
    """

    def __init__(
        self,
        base_channels: int = 32,
        num_blocks=(3, 3, 3),
        num_states: int = 3,
        alpha: float = 0.5,
        inner_lr: float = 1.0,
        smooth: bool = False,
    ):
        super().__init__()

        if len(num_blocks) != 3:
            raise ValueError(
                "num_blocks must contain three values for the "
                "three encoder levels."
            )

        # ----------------------------------------------------
        # Input feature stem
        # ----------------------------------------------------
        self.feature_stem = BasicConv(
            3,
            base_channels,
            kernel_size=3,
            relu=True,
            stride=1,
            bias=False,
            norm=True,
        )

        # ----------------------------------------------------
        # Encoder level 1: H x W
        # ----------------------------------------------------
        self.encoder_level1 = SESGroup(
            embed_dim=base_channels,
            smooth=smooth,
            depths=num_blocks[0],
            num_states=num_states,
            alpha=alpha,
            inner_lr=inner_lr,
        )

        self.downsample1 = BasicConv(
            base_channels,
            base_channels * 2,
            kernel_size=3,
            relu=True,
            stride=2,
            bias=False,
            norm=True,
        )

        # ----------------------------------------------------
        # Encoder level 2: H/2 x W/2
        # ----------------------------------------------------
        self.encoder_level2 = SESGroup(
            embed_dim=base_channels * 2,
            smooth=smooth,
            depths=num_blocks[1],
            num_states=num_states,
            alpha=alpha,
            inner_lr=inner_lr,
        )

        self.downsample2 = BasicConv(
            base_channels * 2,
            base_channels * 4,
            kernel_size=3,
            relu=True,
            stride=2,
            bias=False,
            norm=True,
        )

        # ----------------------------------------------------
        # Encoder level 3: H/4 x W/4
        # ----------------------------------------------------
        self.encoder_level3 = SESGroup(
            embed_dim=base_channels * 4,
            smooth=smooth,
            depths=num_blocks[2],
            num_states=num_states,
            alpha=alpha,
            inner_lr=inner_lr,
        )

        # ----------------------------------------------------
        # Multi-scale shallow context branches
        # ----------------------------------------------------
        self.context_level2 = ShallowContextModule(
            base_channels * 2
        )
        self.context_fusion_level2 = FeatureAggregationModule(
            base_channels * 2
        )

        self.context_level3 = ShallowContextModule(
            base_channels * 4
        )
        self.context_fusion_level3 = FeatureAggregationModule(
            base_channels * 4
        )

    def forward(
        self,
        input_image: torch.Tensor,
    ):
        """
        Args:
            input_image:
                Degraded input image [B, 3, H, W].

        Returns:
            feature_level1:
                Encoder feature [B, C, H, W].

            feature_level2:
                Encoder feature [B, 2C, H/2, W/2].

            feature_level3:
                Encoder feature [B, 4C, H/4, W/4].
        """

        # ----------------------------------------------------
        # Construct multi-scale image inputs
        # ----------------------------------------------------
        image_half = F.interpolate(
            input_image,
            scale_factor=0.5,
        )

        image_quarter = F.interpolate(
            image_half,
            scale_factor=0.5,
        )

        # Shallow context features at H/2 and H/4.
        context_feature_level2 = self.context_level2(image_half)
        context_feature_level3 = self.context_level3(image_quarter)

        # ----------------------------------------------------
        # Encoder level 1
        # ----------------------------------------------------
        feature = self.feature_stem(input_image)

        feature_level1 = self.encoder_level1(feature)

        # ----------------------------------------------------
        # Encoder level 2
        # ----------------------------------------------------
        feature_level2 = self.downsample1(feature_level1)

        feature_level2 = self.context_fusion_level2(
            feature_level2,
            context_feature_level2,
        )

        feature_level2 = self.encoder_level2(feature_level2)

        # ----------------------------------------------------
        # Encoder level 3
        # ----------------------------------------------------
        feature_level3 = self.downsample2(feature_level2)

        feature_level3 = self.context_fusion_level3(
            feature_level3,
            context_feature_level3,
        )

        feature_level3 = self.encoder_level3(feature_level3)

        return (
            feature_level1,
            feature_level2,
            feature_level3,
        )


# ============================================================
# TTTIR Decoder
# ============================================================

class TTTIRDecoder(nn.Module):
    """
    Decoder of TTTIR.

    The decoder progressively reconstructs the restored image from
    coarse to fine. Encoder features are introduced through skip
    connections, and each decoding level is refined by an SESGroup.

    Args:
        base_channels:
            Feature dimension of the finest decoder level.

        num_blocks:
            Number of SES modules at the three decoder levels.

        num_states:
            Number of progressive target states in each SES.

        alpha:
            Spatial-frequency fusion coefficient in PSSG.

        inner_lr:
            Inner-loop learning rate in STE.

        smooth:
            Compatibility argument retained from the original code.
    """

    def __init__(
        self,
        base_channels: int = 32,
        num_blocks=(3, 3, 3),
        num_states: int = 3,
        alpha: float = 0.5,
        inner_lr: float = 1.0,
        smooth: bool = False,
    ):
        super().__init__()

        if len(num_blocks) != 3:
            raise ValueError(
                "num_blocks must contain three values for the "
                "three decoder levels."
            )

        # ----------------------------------------------------
        # Decoder level 3: H/4 x W/4
        # ----------------------------------------------------
        self.decoder_level3 = SESGroup(
            embed_dim=base_channels * 4,
            smooth=smooth,
            depths=num_blocks[2],
            num_states=num_states,
            alpha=alpha,
            inner_lr=inner_lr,
        )

        # H/4 -> H/2
        self.upsample2 = BasicConv(
            base_channels * 4,
            base_channels * 2,
            kernel_size=4,
            relu=True,
            stride=2,
            transpose=True,
            bias=False,
            norm=True,
        )

        # ----------------------------------------------------
        # Decoder level 2: H/2 x W/2
        # ----------------------------------------------------
        self.decoder_level2 = SESGroup(
            embed_dim=base_channels * 2,
            smooth=smooth,
            depths=num_blocks[1],
            num_states=num_states,
            alpha=alpha,
            inner_lr=inner_lr,
        )

        # H/2 -> H
        self.upsample1 = BasicConv(
            base_channels * 2,
            base_channels,
            kernel_size=4,
            relu=True,
            stride=2,
            transpose=True,
            bias=False,
            norm=True,
        )

        # ----------------------------------------------------
        # Decoder level 1: H x W
        # ----------------------------------------------------
        self.decoder_level1 = SESGroup(
            embed_dim=base_channels,
            smooth=smooth,
            depths=num_blocks[0],
            num_states=num_states,
            alpha=alpha,
            inner_lr=inner_lr,
        )

        # ----------------------------------------------------
        # Multi-scale restoration output heads
        # ----------------------------------------------------
        self.output_heads = nn.ModuleList(
            [
                BasicConv(
                    base_channels * 4,
                    3,
                    kernel_size=3,
                    relu=False,
                    stride=1,
                    bias=True,
                ),
                BasicConv(
                    base_channels * 2,
                    3,
                    kernel_size=3,
                    relu=False,
                    stride=1,
                    bias=True,
                ),
                BasicConv(
                    base_channels,
                    3,
                    kernel_size=3,
                    relu=False,
                    stride=1,
                    bias=True,
                ),
            ]
        )

        # ----------------------------------------------------
        # Encoder-decoder skip feature fusion
        # ----------------------------------------------------
        self.skip_fusion_layers = nn.ModuleList(
            [
                BasicConv(
                    base_channels * 4,
                    base_channels * 2,
                    kernel_size=1,
                    relu=True,
                    stride=1,
                ),
                BasicConv(
                    base_channels * 2,
                    base_channels,
                    kernel_size=1,
                    relu=True,
                    stride=1,
                ),
            ]
        )

    def forward(
        self,
        feature_level1: torch.Tensor,
        feature_level2: torch.Tensor,
        feature_level3: torch.Tensor,
        input_image: torch.Tensor,
    ):
        """
        Args:
            feature_level1:
                Encoder feature [B, C, H, W].

            feature_level2:
                Encoder feature [B, 2C, H/2, W/2].

            feature_level3:
                Encoder feature [B, 4C, H/4, W/4].

            input_image:
                Original degraded image [B, 3, H, W].

        Returns:
            outputs:
                Multi-scale restoration outputs ordered from coarse
                to fine:

                    [
                        output_quarter,
                        output_half,
                        output_full,
                    ]
        """

        # ----------------------------------------------------
        # Multi-scale residual images
        # ----------------------------------------------------
        image_half = F.interpolate(
            input_image,
            scale_factor=0.5,
        )

        image_quarter = F.interpolate(
            image_half,
            scale_factor=0.5,
        )

        # ----------------------------------------------------
        # Decoder level 3
        # ----------------------------------------------------
        decoder_feature_level3 = self.decoder_level3(
            feature_level3
        )

        output_quarter = (
            self.output_heads[0](decoder_feature_level3)
            + image_quarter
        )

        # ----------------------------------------------------
        # Decoder level 2
        # ----------------------------------------------------
        decoder_feature_level2 = self.upsample2(
            decoder_feature_level3
        )

        decoder_feature_level2 = torch.cat(
            [
                decoder_feature_level2,
                feature_level2,
            ],
            dim=1,
        )

        decoder_feature_level2 = self.skip_fusion_layers[0](
            decoder_feature_level2
        )

        decoder_feature_level2 = self.decoder_level2(
            decoder_feature_level2
        )

        output_half = (
            self.output_heads[1](decoder_feature_level2)
            + image_half
        )

        # ----------------------------------------------------
        # Decoder level 1
        # ----------------------------------------------------
        decoder_feature_level1 = self.upsample1(
            decoder_feature_level2
        )

        decoder_feature_level1 = torch.cat(
            [
                decoder_feature_level1,
                feature_level1,
            ],
            dim=1,
        )

        decoder_feature_level1 = self.skip_fusion_layers[1](
            decoder_feature_level1
        )

        decoder_feature_level1 = self.decoder_level1(
            decoder_feature_level1
        )

        output_full = (
            self.output_heads[2](decoder_feature_level1)
            + input_image
        )

        outputs = [
            output_quarter,
            output_half,
            output_full,
        ]

        return outputs


# ============================================================
# Complete TTTIR Network
# ============================================================

class TTTIR(nn.Module):
    """
    Complete TTTIR restoration network.

    Architecture:
        input
          -> TTTIREncoder
          -> multi-scale restoration features
          -> TTTIRDecoder
          -> coarse-to-fine restoration outputs

    Args:
        base_channels:
            Base feature dimension. The paper default is 32.

        num_blocks:
            Number of SES modules at the three feature scales.

        num_states:
            Number of progressive target states. For L=2, use 3.

        alpha:
            Spatial-frequency fusion coefficient.

        inner_lr:
            Inner-loop learning rate used by STE.

        smooth:
            Compatibility argument retained from the original code.
    """

    def __init__(
        self,
        base_channels: int = 32,
        num_blocks=(1, 1, 1),
        num_states: int = 3,
        alpha: float = 0.5,
        inner_lr: float = 1.0,
        smooth: bool = False,
    ):
        super().__init__()

        self.encoder = TTTIREncoder(
            base_channels=base_channels,
            num_blocks=num_blocks,
            num_states=num_states,
            alpha=alpha,
            inner_lr=inner_lr,
            smooth=smooth,
        )

        self.decoder = TTTIRDecoder(
            base_channels=base_channels,
            num_blocks=num_blocks,
            num_states=num_states,
            alpha=alpha,
            inner_lr=inner_lr,
            smooth=smooth,
        )

    def forward(
        self,
        input_image: torch.Tensor,
    ):
        feature_level1, feature_level2, feature_level3 = (
            self.encoder(input_image)
        )

        restoration_outputs = self.decoder(
            feature_level1=feature_level1,
            feature_level2=feature_level2,
            feature_level3=feature_level3,
            input_image=input_image,
        )

        return restoration_outputs


# ============================================================
# Network builder
# ============================================================

def build_tttir(
    base_channels: int = 32,
    num_blocks=(1, 1, 1),
    num_states: int = 3,
    alpha: float = 0.5,
    inner_lr: float = 1.0,
    smooth: bool = False,
) -> TTTIR:
    """
    Build the TTTIR network.

    Default configuration:
        base_channels = 32
        num_blocks = (1, 1, 1)
        num_states = 3, corresponding to paper setting L=2
        alpha = 0.5
        inner_lr = 1.0
    """

    return TTTIR(
        base_channels=base_channels,
        num_blocks=num_blocks,
        num_states=num_states,
        alpha=alpha,
        inner_lr=inner_lr,
        smooth=smooth,
    )


# ============================================================
# Backward-compatible entry point
# ============================================================

def build_net() -> TTTIR:
    """
    Compatibility wrapper for training frameworks that still call
    build_net().
    """
    return build_tttir()


# Compatibility alias for checkpoints or scripts importing Model1.
Model1 = TTTIR


# ============================================================
# Complexity test
# ============================================================

if __name__ == "__main__":
    from thop import profile

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    input_tensor = torch.randn(
        1,
        3,
        256,
        256,
        device=device,
    )

    model = build_tttir(
        base_channels=32,
        num_blocks=(1, 1, 1),

        # Paper setting: L=2
        # State sequence: {T_0, T_1, T_2}
        num_states=3,

        alpha=0.5,
        inner_lr=1.0,
    ).to(device)

    flops, params = profile(
        model,
        inputs=(input_tensor,),
    )

    print(
        "flops: %.6f G, params: %.6f M"
        % (
            flops / 1e9,
            params / 1e6,
        )
    )