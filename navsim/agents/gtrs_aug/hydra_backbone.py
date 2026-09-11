# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Vision backbone for GTRS-Aug / DriveSuprim recipe (multi-backbone).
Optional backends (intern/swin/sptr/vit) are imported lazily.
"""

import timm
from torch import nn

from navsim.agents.backbones.vov import VoVNet
from navsim.agents.gtrs_aug.hydra_config_aug import HydraConfigAug


class HydraBackbone(nn.Module):
    """Image encoder with AdaptiveAvgPool to (img_vert_anchors, img_horz_anchors)."""

    def __init__(self, config: HydraConfigAug):
        super().__init__()
        self.config = config
        self.backbone_type = config.backbone_type
        self._is_davit = False

        if config.backbone_type == "intern":
            from navsim.agents.backbones.internimage import InternImage

            self.image_encoder = InternImage(
                init_cfg=dict(type="Pretrained", checkpoint=config.intern_ckpt),
                frozen_stages=2,
            )
            vit_channels = 2560
            self.image_encoder.init_weights()
        elif config.backbone_type == "vov":
            self.image_encoder = VoVNet(
                spec_name="V-99-eSE",
                out_features=["stage4", "stage5"],
                norm_eval=True,
                with_cp=True,
                init_cfg=dict(
                    type="Pretrained",
                    checkpoint=config.vov_ckpt,
                    prefix="img_backbone.",
                ),
            )
            vit_channels = 1024
            self.image_encoder.init_weights()
        elif config.backbone_type == "swin":
            from navsim.agents.backbones.swin import SwinTransformerBEVFT

            self.image_encoder = SwinTransformerBEVFT(
                with_cp=True,
                convert_weights=False,
                depths=[2, 2, 18, 2],
                drop_path_rate=0.35,
                embed_dims=192,
                init_cfg=dict(checkpoint=config.swin_ckpt, type="Pretrained"),
                num_heads=[6, 12, 24, 48],
                out_indices=[3],
                patch_norm=True,
                window_size=[16, 16, 16, 16],
                use_abs_pos_embed=True,
                return_stereo_feat=False,
                output_missing_index_as_none=False,
            )
            vit_channels = 1536
        elif config.backbone_type == "vit":
            from navsim.agents.utils.vit import DAViT

            self.image_encoder = DAViT(ckpt=config.vit_ckpt)
            self._is_davit = True
            vit_channels = 1024
        elif config.backbone_type == "sptr":
            from navsim.agents.backbones.eva import EVAViT

            img_vit_size = (config.camera_height, config.camera_width)
            self.image_encoder = EVAViT(
                img_size=img_vit_size[0],
                patch_size=16,
                window_size=16,
                global_window_size=img_vit_size[0] // 16,
                in_chans=3,
                embed_dim=1024,
                depth=24,
                num_heads=16,
                mlp_ratio=4 * 2 / 3,
                window_block_indexes=(
                    list(range(0, 2))
                    + list(range(3, 5))
                    + list(range(6, 8))
                    + list(range(9, 11))
                    + list(range(12, 14))
                    + list(range(15, 17))
                    + list(range(18, 20))
                    + list(range(21, 23))
                ),
                qkv_bias=True,
                drop_path_rate=0.3,
                with_cp=True,
                flash_attn=False,
                xformers_attn=True,
            )
            self.image_encoder.init_weights(config.sptr_ckpt)
            vit_channels = 1024
        elif config.backbone_type == "resnet34":
            self.image_encoder = timm.create_model("resnet34", pretrained=False, features_only=True)
            vit_channels = 512
        elif config.backbone_type == "resnet50":
            self.image_encoder = timm.create_model("resnet50", pretrained=False, features_only=True)
            vit_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone_type={config.backbone_type}")

        self.avgpool_img = nn.AdaptiveAvgPool2d((self.config.img_vert_anchors, self.config.img_horz_anchors))
        self.img_feat_c = vit_channels

    def _encode(self, image, **kwargs):
        if self._is_davit:
            image_feat = self.image_encoder(image, **kwargs)[-1]
        else:
            image_feat = self.image_encoder(image)[-1]
        return image_feat

    def forward(self, image, **kwargs):
        image_feat = self._encode(image, **kwargs)
        return self.avgpool_img(image_feat)

    def forward_tup(self, image, **kwargs):
        image_feat = self._encode(image, **kwargs)
        class_feat = image_feat.mean(dim=(-1, -2))
        pooled = self.avgpool_img(image_feat)
        if self.config.lab.use_higher_res_feat_in_refinement:
            return (pooled, class_feat, image_feat)
        return (pooled, class_feat)
