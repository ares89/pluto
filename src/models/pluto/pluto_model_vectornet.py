from torch import Size, Tensor, nn
from typing import List, Union
import numbers
from copy import deepcopy
import math

import torch
from torch import Tensor
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from src.feature_builders.pluto_feature_builder import PlutoFeatureBuilder

from .layers.fourier_embedding import FourierEmbedding
from .layers.transformer import TransformerEncoderLayer
from .modules.agent_encoder import AgentEncoder
from .modules.agent_predictor import AgentPredictor
from .modules.map_encoder import MapEncoder
from .modules.static_objects_encoder import StaticObjectsEncoder
from .modules.planning_decoder import PlanningDecoder
from .layers.mlp_layer import MLPLayer

# no meaning, required by nuplan
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)


class LayerNorm(nn.LayerNorm):
    weight: Tensor
    bias: Tensor

    """
    Layer normalization with single dimention norm support.
    Only support NCHW input layout.

    Args:
        normalized_shape (int):
            Input shape from an expected input of size
            [* x normalized_shape[0] x … x normalized_shape[-1]].
        eps (float, optional):
            A value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool, optional):
            Whether use learnable per-element affine parameters.
            Defaults to True.
        dim (int, optional):
            If specified, the normalization will be done alone this dimention.
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
        dim=None,
    ):
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        assert isinstance(
            normalized_shape, (list, tuple, Size)
        ), "normalized_shape muse be a intergral or list or tuple or torch.Size"  # noqa: E501
        assert (
            len(normalized_shape) < 4
        ), "Only support layernorm on W or HW or CHW."
        for v in normalized_shape:
            assert isinstance(
                v, numbers.Integral
            ), "elements of normalized_shape must be integral"
        assert isinstance(eps, float), "param eps must be a float"
        assert isinstance(
            elementwise_affine, bool
        ), "param elementwise_affine must be a bool"
        assert isinstance(
            dim, (type(None), numbers.Integral)
        ), "param dim must be None or a integral"
        assert dim in (
            None,
            1,
            2,
            3,
            -1,
            -2,
            -3,
        ), "Only support layernorm on W or HW or CHW."

        if dim is not None:
            if dim < 0:
                assert (
                    len(normalized_shape) == -dim
                ), "normalized_shape should not include dim before the dim to be normalized"  # noqa: E501
            normalized_shape = [normalized_shape[0]] + (
                [1] * (len(normalized_shape) - 1)
            )

        super(LayerNorm, self).__init__(
            normalized_shape, eps, elementwise_affine, device, dtype
        )

        self.dim = dim

    def single_dim_norm(self, input: torch.Tensor):
        if self.dim >= 0:
            assert (
                self.dim + len(self.normalized_shape) == input.ndim
            ), "normalized_shape should not include dim before the dim to be normalized"  # noqa: E501

        var, mean = torch.var_mean(
            input, dim=self.dim, unbiased=False, keepdim=True
        )
        x = (input - mean) * torch.rsqrt(var + self.eps)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        return x

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input):
        input = input.to(torch.float32)
        if self.dim is not None:
            return self.single_dim_norm(input)
        else:
            return super(LayerNorm, self).forward(input)


class SubGraphLayer(nn.Module):
    """Implements the vectornet subgraph layer.

    Args:
        in_channels: input channels.
        hidden_size: hidden_size.
        out_channels: output channels.
        num_vec: number of vectors.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        out_channels: int,
        num_vec: int,
    ):
        super(SubGraphLayer, self).__init__()
        hidden_size = out_channels
        self.mlp = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            LayerNorm(normalized_shape=[hidden_size, 1, 1], dim=1),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Perform forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = self.mlp(x)
        return x


class SubGraph(nn.Module):
    """Implements the vectornet subgraph.

    Args:
        in_channels: input channels.
        depth: depth for encoder layer.
        hidden_size: hidden_size.
        num_vec: number of vectors.
    """

    def __init__(self, in_channels, depth=3, hidden_size=64, num_vec=9):
        super(SubGraph, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                SubGraphLayer(
                    in_channels=in_channels if i == 0 else hidden_size,
                    hidden_size=hidden_size,
                    out_channels=hidden_size,
                    num_vec=num_vec,
                )
            )
        self.max_pool = nn.MaxPool2d([num_vec, 1], stride=1)

    def forward(self, x: Tensor) -> Tensor:
        """Perform forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        x = self.max_pool(x)
        return x


class PlanningVectornetModel(TorchModuleWrapper):
    def __init__(
        self,
        dim=128,
        state_channel=6,
        polygon_channel=6,
        history_channel=9,
        history_steps=21,
        future_steps=80,
        encoder_depth=4,
        decoder_depth=4,
        drop_path=0.2,
        dropout=0.1,
        num_heads=8,
        num_modes=6,
        use_ego_history=False,
        state_attn_encoder=True,
        state_dropout=0.75,
        use_hidden_proj=False,
        cat_x=False,
        ref_free_traj=False,
        feature_builder: PlutoFeatureBuilder = PlutoFeatureBuilder(),
    ) -> None:
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )

        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.use_hidden_proj = use_hidden_proj
        self.num_modes = num_modes
        self.radius = feature_builder.radius
        self.ref_free_traj = ref_free_traj

        self.pos_emb = FourierEmbedding(3, dim, 64)

        self.lane_enc = SubGraph(in_channels=5, num_vec=19, depth=3, hidden_size=dim)  # max_lane:256
        self.traj_enc = SubGraph(in_channels=5, num_vec=20, depth=3, hidden_size=dim)  # max_agent:64

        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )
        self.norm = nn.LayerNorm(dim)

        self.agent_predictor = AgentPredictor(dim=dim, future_steps=future_steps)
        self.planning_decoder = PlanningDecoder(
            num_mode=num_modes,
            decoder_depth=decoder_depth,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=4,
            dropout=dropout,
            cat_x=cat_x,
            future_steps=future_steps,
        )

        if use_hidden_proj:
            self.hidden_proj = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
            )

        if self.ref_free_traj:
            self.ref_free_decoder = MLPLayer(dim, 2 * dim, future_steps * 4)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, data):
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]

        max_agent = 64
        max_lane = 256
        tmp_traj_feat = torch.cat([data['agent']['position'][:, :, : self.history_steps-1, :2],
                                   data['agent']['position'][:, :, 1: self.history_steps, :2],
                                   torch.linalg.norm(data['agent']['velocity'][:, :, 1: self.history_steps, :2], dim=-1, keepdim=True)
                                   - torch.linalg.norm(data['agent']['velocity'][:, :, 0: self.history_steps-1, :2], dim=-1, keepdim=True)], dim=-1)
        tmp_traj_feat = tmp_traj_feat*(agent_mask[:, :, :-1] & agent_mask[:, :, 1:]).unsqueeze(-1)
        traj_feat = torch.zeros(tmp_traj_feat.shape[0], max_agent, tmp_traj_feat.shape[2], tmp_traj_feat.shape[3], device=agent_pos.device)
        traj_feat[:, :tmp_traj_feat.shape[1]] = tmp_traj_feat

        pad_agent_mask = torch.zeros(tmp_traj_feat.shape[0], max_agent, tmp_traj_feat.shape[2], device=agent_pos.device)
        pad_agent_mask[:, :tmp_traj_feat.shape[1]] = agent_mask[:, :, :-1] & agent_mask[:, :, 1:]

        tmp_lane_feat = torch.cat([
            data["map"]['point_position'][:, :, 0, :19],
            data["map"]['point_position'][:, :, 0, 1:],
            data["map"]['polygon_type'].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 19, 1)
        ], dim=-1)
        tmp_lane_feat = tmp_lane_feat*(polygon_mask[:, :, :-1] & polygon_mask[:, :, 1:]).unsqueeze(-1)
        lane_feat = torch.zeros(tmp_lane_feat.shape[0], max_lane, tmp_lane_feat.shape[2], tmp_lane_feat.shape[3], device=agent_pos.device)
        lane_feat[:, :tmp_lane_feat.shape[1]] = tmp_lane_feat

        pad_lane_mask = torch.zeros(tmp_lane_feat.shape[0], max_lane, tmp_lane_feat.shape[2], device=agent_pos.device)
        pad_lane_mask[:, :tmp_lane_feat.shape[1]] = polygon_mask[:, :, :-1] & polygon_mask[:, :, 1:]

        instance_mask = torch.cat([pad_agent_mask.sum(-1) > 0, pad_lane_mask.sum(-1) > 0], dim=-1)  # torch.Size([4, 320])

        position = torch.cat([traj_feat[:, :, -1, :2], lane_feat[:, :, 9, :2]], dim=1)
        last_agent_vec = traj_feat[:, :, -1, 2:4]-traj_feat[:, :, -1, :2]
        agent_heading = torch.atan2(last_agent_vec[..., -1], last_agent_vec[..., 0])
        lane_vec = lane_feat[:, :, 9, 2:4]-lane_feat[:, :, 9, :2]
        lane_heading = torch.atan2(lane_vec[..., -1], lane_vec[..., 0])

        angle = torch.cat([agent_heading, lane_heading], dim=1)
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        pos = torch.cat([position, angle.unsqueeze(-1)], dim=-1)

        key_padding_mask = instance_mask  # torch.Size([4, 320])

        x_agent = self.traj_enc(traj_feat.permute(0, 3, 2, 1))
        x_polygon = self.lane_enc(lane_feat.permute(0, 3, 2, 1))

        x = torch.cat([x_agent, x_polygon], dim=-1)  # torch.Size([4, 128, 1, 320])
        pos_embed = self.pos_emb(pos)
        x = x.squeeze().permute(0, 2, 1) + pos_embed

        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask, return_attn_weights=False)
        x = self.norm(x)

        prediction = self.agent_predictor(x[:, 1:max_agent])

        ref_line_available = data["reference_line"]["position"].shape[1] > 0

        if ref_line_available:
            trajectory, probability = self.planning_decoder(
                data, {"enc_emb": x, "enc_key_padding_mask": key_padding_mask}
            )
        else:
            trajectory, probability = None, None

        out = {
            "trajectory": trajectory,
            "probability": probability,  # (bs, R, M)
            "prediction": prediction[:,1:tmp_traj_feat.shape[1]],  # (bs, A-1, T, 2)
        }

        if self.use_hidden_proj:
            out["hidden"] = self.hidden_proj(x[:, 0])

        if self.ref_free_traj:
            ref_free_traj = self.ref_free_decoder(x[:, 0]).reshape(
                bs, self.future_steps, 4
            )
            out["ref_free_trajectory"] = ref_free_traj

        if not self.training:
            if self.ref_free_traj:
                ref_free_traj_angle = torch.arctan2(
                    ref_free_traj[..., 3], ref_free_traj[..., 2]
                )
                ref_free_traj = torch.cat(
                    [ref_free_traj[..., :2], ref_free_traj_angle.unsqueeze(-1)], dim=-1
                )
                out["output_ref_free_trajectory"] = ref_free_traj

            output_prediction = torch.cat(
                [
                    prediction[..., :2] + position[:, 1:max_agent, None],
                    torch.atan2(prediction[..., 3], prediction[..., 2]).unsqueeze(-1)
                    + agent_heading[:, 1:max_agent, None, None],
                    prediction[..., 4:6],
                ],
                dim=-1,
            )
            out["output_prediction"] = output_prediction

            if trajectory is not None:
                r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)
                probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

                angle = torch.atan2(trajectory[..., 3], trajectory[..., 2])
                out_trajectory = torch.cat(
                    [trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
                )

                bs, R, M, T, _ = out_trajectory.shape
                flattened_probability = probability.reshape(bs, R * M)
                best_trajectory = out_trajectory.reshape(bs, R * M, T, -1)[
                    torch.arange(bs), flattened_probability.argmax(-1)
                ]

                out["output_trajectory"] = best_trajectory
                out["candidate_trajectories"] = out_trajectory
            else:
                out["output_trajectory"] = out["output_ref_free_trajectory"]
                out["probability"] = torch.zeros(1, 0, 0)
                out["candidate_trajectories"] = torch.zeros(
                    1, 0, 0, self.future_steps, 3
                )

        return out
