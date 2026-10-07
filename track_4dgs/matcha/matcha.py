# Code adapted from https://github.com/nv-dvl/matcha/blob/master/matcha/model/matcha.py
from functools import partial
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from .dust3r.blocks import DecoderBlockJointP
from .dust3r.patch_embed import get_patch_embed
from .dust3r.pos_embed import RoPE2D
from .geoaware_sc.corr_map_model import Correlation2Displacement
from .dift.dift_sd import DIFT
from .dinov2 import DINOv2
from .utils.category_list import cats_spair, cats_pascal, cats_ap10k, cats_willow


class BasicLayer(nn.Module):
    """
    Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False,
    ):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


def concat_feature(feat_c, feat_f):
    if feat_f.shape[2] != feat_c.shape[2] or feat_f.shape[3] != feat_c.shape[3]:
        feat_c = F.interpolate(
            feat_c, size=(feat_f.shape[2], feat_f.shape[3]), mode="bilinear"
        )
        feat_c = F.normalize(feat_c, dim=1)

    stride = feat_c.shape[1] // feat_f.shape[1]
    feat = torch.concat([feat_f, feat_c[:, ::stride, :, :]], dim=1)
    feat = F.normalize(feat, dim=1)
    return feat


class AttentionFusionNet(nn.Module):
    def __init__(
            self,
            image_size=512,
            ft_dim_c=1280,
            ft_dim_f=640,
            out_dim_c=768,
            out_dim_f=256,
            ensemble_size=8,
            hidden_dim=512,
            patch_size_c=2,
            patch_size_f=2,
            patch_embed_cls="PatchEmbedDust3R",
            dec_depth=8,
            dec_num_heads=8,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            norm_im2_in_dec=True,
            pos_embed="cosine",
            use_corr=True,
            **kwargs,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_embed_cls = patch_embed_cls
        self.patch_size_c = patch_size_c
        self.patch_size_f = patch_size_f
        self.dec_depth = dec_depth
        self.ensemble_size = ensemble_size

        # Keypoint head
        self.keypoint_encoder = BasicLayer(3, 16, 3, padding=1)
        self.keypoint_head = nn.Sequential(
            BasicLayer(16 * 64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 65, 1),
        )
        self.heatmap_head = nn.Sequential(
            BasicLayer(out_dim_f, 64, 3, padding=1),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

        if pos_embed == "cosine":
            self.rope = None  # nothing for cosine
        elif pos_embed.startswith("RoPE"):  # eg RoPE100
            self.enc_pos_embed = None  # nothing to add in the encoder with RoPE
            self.dec_pos_embed = None  # nothing to add in the decoder with RoPE
            freq = float(pos_embed[len("RoPE"):])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError("Unknown pos_embed " + pos_embed)

        self.patch_embed_c = get_patch_embed(
            self.patch_embed_cls,
            img_size=self.image_size // 8,
            patch_size=self.patch_size_c,
            in_chans=ft_dim_c,
            enc_embed_dim=hidden_dim,
        )
        self.patch_embed_f = get_patch_embed(
            self.patch_embed_cls,
            img_size=self.image_size // 8,
            patch_size=self.patch_size_f,
            in_chans=ft_dim_f,
            enc_embed_dim=hidden_dim,
        )

        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlockJointP(
                    hidden_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                    only_geo=False,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm_c = norm_layer(hidden_dim)
        self.dec_norm_f = norm_layer(hidden_dim)
        self.fusion_c = nn.Sequential(
            BasicLayer(ft_dim_c + hidden_dim // patch_size_c ** 2, 1024, 3, padding=1),
            BasicLayer(1024, 1024, 3, padding=1),
            nn.Conv2d(1024, out_dim_c, kernel_size=1, padding=0),
        )
        self.fusion_f = nn.Sequential(
            BasicLayer(ft_dim_f + hidden_dim // patch_size_f ** 2, 1024, 3, padding=1),
            BasicLayer(1024, 1024, 3, padding=1),
            nn.Conv2d(1024, out_dim_f, kernel_size=1, padding=0),
        )

        self.fine_matcher = nn.Sequential(
            nn.Linear(out_dim_f * 2, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
        )

        if use_corr:
            self.corr_model = Correlation2Displacement(setting=1)
        else:
            self.corr_model = None

    def forward_keypoints(self, img: torch.Tensor, ws=8):
        kp_feat = self.keypoint_encoder(img)

        # Unfolds tensor in 2D with desired ws (window size) and concat the channels
        B, C, H, W = kp_feat.shape
        kp_feat = (
            kp_feat.unfold(2, ws, ws)
            .unfold(3, ws, ws)
            .reshape(B, C, H // ws, W // ws, ws ** 2)
        )
        kp_feat = kp_feat.permute(0, 1, 4, 2, 3).reshape(B, -1, H // ws, W // ws)
        keypoints = self.keypoint_head(kp_feat)
        return keypoints

    def fuse_semantic(self, feat_c: torch.Tensor, feat_f: torch.Tensor):
        # Upsample coarse feature map to the same size as fine feature
        B, _, H, W = feat_f.shape
        feat_c = F.interpolate(feat_c, size=(H, W), mode="bilinear")

        feat_c_patch, pos_c = self.patch_embed_c(feat_c)
        feat_f_patch, pos_f = self.patch_embed_f(feat_f)

        for blk in self.dec_blocks:
            # order - bc some params will not be used
            feat_f_patch, feat_c_patch = blk(feat_f_patch, feat_c_patch, pos_f, pos_c)

        feat_c_att = self.dec_norm_c(feat_c_patch)
        feat_c_att = feat_c_att.transpose(-1, -2).view(
            B, -1, H // self.patch_size_c, W // self.patch_size_c
        )

        # [B, C, H // patch_size, W // patch_size] - > [B, C // patch_size ** 2, H, W]
        feat_c_att = F.pixel_shuffle(feat_c_att, self.patch_size_c)
        feat_c_final = torch.cat([feat_c, feat_c_att], dim=1)
        feats_c = self.fusion_c(feat_c_final)
        feats_c = F.normalize(feats_c, dim=1)

        return feats_c, feat_f_patch

    def forward_fuse_feature(self, feat_c: torch.Tensor, feat_f: torch.Tensor):
        # Semantic feature attention
        feats_c, feat_f_patch = self.fuse_semantic(feat_c=feat_c, feat_f=feat_f)

        # Geometric feature attention
        B, _, H, W = feat_f.shape
        feat_f_att = self.dec_norm_f(feat_f_patch)
        feat_f_att = feat_f_att.transpose(-1, -2).view(
            B, -1, H // self.patch_size_f, W // self.patch_size_f
        )
        # [B, C, H // patch_size, W // patch_size] - > [B, C // patch_size ** 2, H, W]
        feat_f_att = F.pixel_shuffle(feat_f_att, self.patch_size_f)
        feat_f_final = torch.cat([feat_f, feat_f_att], dim=1)
        feats_f = self.fusion_f(feat_f_final)
        heatmap = self.heatmap_head(feats_f)
        return feats_c, feats_f, heatmap

    def forward(self, x: torch.Tensor, feat_c: torch.Tensor, feat_f: torch.Tensor):
        feats_c, feats_f, heatmap = self.forward_fuse_feature(
            feat_c=feat_c, feat_f=feat_f
        )
        keypoints = self.forward_keypoints(x)
        return feats_c, feats_f, heatmap, keypoints


class MatchaLight(AttentionFusionNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dift = DIFT(
            cat_list=cats_pascal + cats_willow + cats_ap10k + cats_spair
        )

        self.name = "MatchaLight"

    def forward(self, img: torch.Tensor, cat=None, feat_c=None, feat_f=None, **kwargs):
        # Extract raw dift features
        if feat_c is None or feat_f is None:
            feat_c, feat_f = self.dift(img, cat=cat, ensemble_size=self.ensemble_size)

        # Feature fusion
        feats_c, feats_f, _ = self.forward_fuse_feature(feat_c, feat_f)
        feats_f = F.normalize(feats_f, dim=1)
        feats_c = F.normalize(feats_c, dim=1)
        return feats_c, feats_f

    def forward_unified(
            self,
            img: torch.Tensor,
            feat_c=None,
            feat_f=None,
    ):
        feats_c, feats_f = self.forward(img, feat_c, feat_f)
        feat_uni = concat_feature(feat_c=feats_c, feat_f=feats_f)
        return feat_uni

    def runtime_benchmark(self, img: torch.Tensor):
        timer = {}

        torch.cuda.synchronize()
        start = time.time()
        # Extract raw dift features
        feat_c, feat_f = self.dift(img, ensemble_size=self.ensemble_size)
        torch.cuda.synchronize()
        end = time.time()
        runtime = end - start
        timer["DIFT"] = runtime

        torch.cuda.synchronize()
        start = time.time()
        # DIFT feature fusion
        feat_s, feat_g, _ = self.forward_fuse_feature(feat_c, feat_f)
        feat_g = F.normalize(feat_g, dim=1)
        feat_s = F.normalize(feat_s, dim=1)
        torch.cuda.synchronize()
        end = time.time()
        runtime = end - start
        timer["Fuse"] = runtime
        return timer


class Matcha(AttentionFusionNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Load DIFT
        self.dift = DIFT(
            cat_list=cats_pascal + cats_willow + cats_ap10k + cats_spair
        )

        # Load DINOv2
        self.dinov2 = DINOv2()
        self.name = "Matcha"

    def forward(self, img: torch.Tensor, feat_c=None, feat_f=None, cat=None, semantic_mode=False):
        # Extract raw dift features
        if feat_c is None or feat_f is None:
            feat_c, feat_f = self.dift(img, cat=cat, ensemble_size=self.ensemble_size)

        # DIFT feature fusion
        feat_s, feat_g, _ = self.forward_fuse_feature(feat_c=feat_c, feat_f=feat_f)
        feat_g = F.normalize(feat_g, dim=1)
        feat_s = F.normalize(feat_s, dim=1)

        # Extract raw dino features
        feat_dino = self.dinov2(img)
        if semantic_mode:
            feat = concat_feature(
                feat_f=concat_feature(feat_f=feat_g, feat_c=feat_dino),
                feat_c=feat_s,
            )
        else:
            feat = concat_feature(
                feat_f=concat_feature(feat_f=feat_g, feat_c=feat_s),
                feat_c=feat_dino,
            )
        return feat

    def runtime_benchmark(self, img: torch.Tensor):
        timer = {}

        torch.cuda.synchronize()
        start = time.time()
        # Extract raw dift features
        feat_c, feat_f = self.dift(img, ensemble_size=self.ensemble_size)
        torch.cuda.synchronize()
        end = time.time()
        runtime = end - start
        timer["DIFT"] = runtime

        torch.cuda.synchronize()
        start = time.time()
        # DIFT feature fusion
        feat_s, feat_g, _ = self.forward_fuse_feature(feat_c, feat_f)
        feat_g = F.normalize(feat_g, dim=1)
        feat_s = F.normalize(feat_s, dim=1)
        torch.cuda.synchronize()
        end = time.time()
        runtime = end - start
        timer["Fuse"] = runtime

        torch.cuda.synchronize()
        start = time.time()
        # Extract raw dino features
        dino_feat = self.dinov2(img)
        torch.cuda.synchronize()
        end = time.time()
        runtime = end - start
        timer["DINOv2"] = runtime

        torch.cuda.synchronize()
        start = time.time()
        feat = concat_feature(
            feat_f=concat_feature(feat_f=feat_g, feat_c=feat_s),
            feat_c=dino_feat,
        )
        torch.cuda.synchronize()
        end = time.time()
        runtime = end - start
        timer["Merge"] = runtime

        return timer