import torch
import torch.nn as nn
import torch.nn.functional as F

from model.CNNBlock import (
    ConvGNAct, DownsampleBlock, StemLayer, PatchEmbed,
    make_group_norm, CrackOrientedBlock,
)
from model.FusionBlock import CNNandMambaBlock


# ==================== Decoder Block ====================

class DecoderBlock(nn.Module):
    """解码器块: Upsample → Gated Skip Fusion → Refine"""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            make_group_norm(out_channels),
            nn.GELU(),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, 1, bias=False),
            make_group_norm(out_channels),
            nn.GELU(),
        )
        self.skip_gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            make_group_norm(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            make_group_norm(out_channels),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            make_group_norm(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            make_group_norm(out_channels),
        )
        self.refine_act = nn.GELU()

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        x = self.upsample(x)
        skip = self.skip_proj(skip)
        gate = self.skip_gate(torch.cat([x, skip], dim=1))
        x = self.fuse(torch.cat([x, skip * gate], dim=1))
        x = self.refine_act(x + self.refine(x))
        return x


# ==================== Main Network ====================

class TriBranchCrackNet(nn.Module):
    """
    CNN-Mamba 裂缝分割网络

    Stem → 6 Stage Encoder → Decoder

    Encoder:
      Stage 0-1: CrackOrientedBlock — 纯 CNN 局部特征
      Stage 2-5: CNNandMambaBlock — CNN ∥ Mamba 交叉注意力

    Decoder:
      DecoderBlock (Upsample + Gated Skip Fusion + Edge-Guided Refine)
      Full-res Refiner (原图分辨率细化，缓解细裂缝稀释)
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        widths=(24, 48, 96, 192),
        stage_blocks=None,
        k=7,
        d_state=16,
        dropout=0.1,
        attn_drop=0.0,
        proj_drop=0.1,
        drop_path_rate=0.1,
        first_downsample_stride: int = 1,
        aux_stage_indices=(),
        use_aux_outputs=False,
        use_edge_head=False,
        **kwargs,
    ):
        super().__init__()
        if stage_blocks is None:
            stage_blocks = (1, 1, 1, 1)
        if len(stage_blocks) != len(widths):
            raise ValueError("'widths' and stage_blocks must have the same length")
        if len(widths) < 3 or len(widths) > 6:
            raise ValueError("This layout expects 3 to 6 stages")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.widths = widths
        self.stage_blocks = stage_blocks
        self.stage_count = len(widths)
        self.aux_stage_indices = tuple(aux_stage_indices or ())
        self.use_aux_outputs = use_aux_outputs or use_edge_head
        self.use_edge_head = use_edge_head
        self.window_size = max(1, int(kwargs.get("window_size", 7)))
        self.global_pool = max(1, int(kwargs.get("global_pool", 4)))
        self.scan_directions = int(kwargs.get("scan_directions", 8))

        num_aux_stages = max(self.stage_count - 1, 0)
        if any(idx < 0 or idx >= num_aux_stages for idx in self.aux_stage_indices):
            raise ValueError(f"aux_stage_indices out of range: {self.aux_stage_indices}")

        # ===================== Stem (仅下采样) =====================
        stem_dims0 = widths[0] * 2  # Stem outputs widths[0]=48ch, 保留更多早期特征
        self.stem = StemLayer(
            in_channels=in_channels,
            dims=[stem_dims0]
        )
        stem_channels = self.stem.out_channels

        # ===================== Encoder =====================
        # Stage 0-1: CrackOrientedBlock (CNN only)
        # Stage 2-5: CNNandMambaBlock (后 4 个 stage 全用 Mamba)
        mamba_start = 2  # 从 Stage 2 (索引 2) 开始用 Mamba

        self.stage_downs = nn.ModuleList()
        self.stages = nn.ModuleList()
        total_blocks = sum(stage_blocks)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist() if total_blocks > 0 else []
        block_ptr = 0

        for idx, (width, blocks) in enumerate(zip(widths, stage_blocks)):
            # 下采样层
            if idx == 0:
                self.stage_downs.append(
                    PatchEmbed(
                        in_channels=stem_channels,
                        embed_dim=width,
                        patch_size=3,
                        stride=first_downsample_stride,
                        padding=1
                    )
                )
            else:
                self.stage_downs.append(DownsampleBlock(widths[idx - 1], width, stride=2))

            # 构建阶段块
            stage_block_list = []
            for _ in range(blocks):
                current_drop_path = dpr[block_ptr] if block_ptr < len(dpr) else 0.0
                block_ptr += 1
                if idx >= mamba_start:
                    # 除最后一个 stage 外用多尺度 Mamba
                    use_ms = (idx < len(widths) - 1)
                    block = CNNandMambaBlock(
                        channels=width, k=k,
                        d_state=d_state,
                        attn_drop=attn_drop, proj_drop=proj_drop,
                        drop=dropout,
                        drop_path=current_drop_path,
                        kv_pool_ratio=self.global_pool,
                        scan_directions=self.scan_directions,
                        use_multiscale_mamba=use_ms,
                    )
                else:
                    # Stage 0-1: 纯 CNN
                    block = CrackOrientedBlock(channels=width, k=k)

                stage_block_list.append(block)
            self.stages.append(nn.Sequential(*stage_block_list))

        # ===================== Decoder =====================
        self.dec_blocks = nn.ModuleList()
        self.aux_heads = nn.ModuleList()

        for idx in range(self.stage_count - 1):
            in_ch = widths[-1 - idx]
            out_ch = widths[-2 - idx]
            skip_ch = out_ch

            self.dec_blocks.append(
                DecoderBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                )
            )

            self.aux_heads.append(
                nn.Sequential(
                    ConvGNAct(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(out_ch, num_classes, kernel_size=1, stride=1, padding=0),
                )
            )

        # 最终细化: decoder 输出 + stem 浅层特征 → 精修细裂缝边界
        # stem_out 包含仅经过一次 stride-2 的浅层边缘信息，直接 concat 可恢复细节
        refine_in_ch = widths[0] + stem_channels  # decoder_out + stem_skip
        refine_ch = widths[0] * 2
        self.full_res_refine = nn.Sequential(
            ConvGNAct(refine_in_ch, refine_ch, kernel_size=3, stride=1, padding=1),
            ConvGNAct(refine_ch, widths[0], kernel_size=3, stride=1, padding=1),
        )

        # ===================== 分割头 =====================
        self.seg_head = nn.Sequential(
            ConvGNAct(widths[0], widths[0], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(widths[0], num_classes, kernel_size=1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        orig_h, orig_w = x.shape[2], x.shape[3]

        # Stem
        stem_out = self.stem(x)

        # Encoder
        stage_feats = []
        y = stem_out
        for down, stage in zip(self.stage_downs, self.stages):
            y = stage(down(y))
            stage_feats.append(y)

        # Decoder
        y = stage_feats[-1]
        aux_outputs = {}
        for idx in range(self.stage_count - 1):
            skip = stage_feats[-2 - idx]
            y = self.dec_blocks[idx](y, skip)

            if idx in self.aux_stage_indices:
                aux_outputs[idx] = self.aux_heads[idx](y)

        # 全分辨率细化: 拼接 stem 浅层特征恢复细裂缝边界
        stem_up = F.interpolate(stem_out, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        y = F.interpolate(y, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        y = torch.cat([y, stem_up], dim=1)
        y = self.full_res_refine(y)

        # 分割输出
        logits = self.seg_head(y)

        if logits.shape[-2:] != (orig_h, orig_w):
            logits = F.interpolate(logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        if self.use_aux_outputs:
            aux_upsampled = {
                idx: F.interpolate(aux, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                for idx, aux in aux_outputs.items()
            }
            return {
                'main_output': logits,
                'aux_outputs': aux_upsampled,
            }

        return logits

    def get_params_groups(self, lr, weight_decay=1e-4):
        bn_types = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.GroupNorm,
        )

        module_dict = dict(self.named_modules())
        no_decay = []
        normal_decay = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            module_name = name.rsplit('.', 1)[0] if '.' in name else ''
            owning_module = module_dict.get(module_name, None)

            if name.endswith('.bias'):
                no_decay.append(param)
            elif isinstance(owning_module, bn_types):
                no_decay.append(param)
            else:
                normal_decay.append(param)

        return [
            {'params': normal_decay, 'lr': lr, 'weight_decay': weight_decay},
            {'params': no_decay, 'lr': lr, 'weight_decay': 0.0}
        ]
