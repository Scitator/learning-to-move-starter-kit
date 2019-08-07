from typing import List
from copy import deepcopy

import torch
import torch.nn as nn

from catalyst import utils


def _get_convolution_net(
    in_channels: int,
    history_len: int = 1,
    channels: List = None,
    use_bias: bool = False,
    use_groups: bool = False,
    use_normalization: bool = False,
    use_dropout: bool = False,
    activation: str = "ReLU"
) -> nn.Module:

    channels = channels or [16, 32, 16]
    activation_fn = torch.nn.__dict__[activation]

    def _get_block(**conv_params):
        layers = []
        layers.append(nn.Conv2d(**conv_params))
        if use_normalization:
            layers.append(nn.InstanceNorm2d(conv_params["out_channels"]))
        if use_dropout:
            layers.append(nn.Dropout2d(p=0.1))
        layers.append(activation_fn(inplace=True))
        return layers

    channels.insert(0, history_len * in_channels)
    params = []
    for i, (in_channels, out_channels) in enumerate(utils.pairwise(channels)):
        num_groups = 1
        if use_groups:
            num_groups = history_len if i == 0 else 4
        params.append(
            {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "bias": use_bias,
                "kernel_size": 3,
                "stride": 1,
                "groups": num_groups,
            }
        )

    layers = []
    for block_params in params:
        layers.extend(_get_block(**block_params))

    net = nn.Sequential(*layers)
    net.apply(utils.create_optimal_inner_init(activation_fn))

    # input_shape: tuple = (2, 11, 11)
    # conv_input = torch.Tensor(torch.randn((1,) + input_shape))
    # conv_output = net(conv_input)
    # print(conv_output.shape, conv_output.nelement())
    # torch.Size([1, 16, 5, 5]) 400

    return net


def _get_linear_net(
    in_features: int,
    history_len: int = 1,
    features: List = None,
    use_bias: bool = False,
    use_normalization: bool = False,
    use_dropout: bool = False,
    activation: str = "ReLU"
) -> nn.Module:

    features = features or [64, 128, 64]
    activation_fn = torch.nn.__dict__[activation]

    def _get_block(**linear_params):
        layers = [nn.Linear(**linear_params)]
        if use_normalization:
            layers.append(nn.LayerNorm(linear_params["out_features"]))
        if use_dropout:
            layers.append(nn.Dropout(p=0.1))
        layers.append(activation_fn(inplace=True))
        return layers

    features.insert(0, history_len * in_features)
    params = []
    for i, (in_features, out_features) in enumerate(utils.pairwise(features)):
        params.append(
            {
                "in_features": in_features,
                "out_features": out_features,
                "bias": use_bias,
            }
        )

    layers = []
    for block_params in params:
        layers.extend(_get_block(**block_params))

    net = nn.Sequential(*layers)
    net.apply(utils.create_optimal_inner_init(activation_fn))

    return net


class StateNet(nn.Module):
    def __init__(
        self,
        main_net: nn.Module,
        features_net: nn.Module = None,
        vector_field_net: nn.Module = None,
    ):
        super().__init__()
        self.main_net = main_net
        self.features_net = features_net
        self.vector_field_net = vector_field_net

    def forward(self, state):
        features = state["features"]
        vector_field = state["vector_field"]

        batch_size, _, _ = features.shape
        features = features.contiguous().view(batch_size, -1)
        features = self.features_net(features)
        features = features.contiguous().view(batch_size, -1)

        batch_size, _, _, h, w = vector_field.shape
        vector_field = vector_field.contiguous().view(batch_size, -1, h, w)
        vector_field = self.vector_field_net(vector_field)
        vector_field = vector_field.contiguous().view(batch_size, -1)

        x = torch.cat([features, vector_field], dim=1)
        x = self.main_net(x)

        return x

    @classmethod
    def get_from_params(
        cls,
        features_net_params=None,
        vector_field_net_params=None,
        # aggregation_net_params=None,
        main_net_params=None,
    ) -> "StateNet":
        features_net_params = deepcopy(features_net_params)
        vector_field_net_params = deepcopy(vector_field_net_params)
        main_net_params = deepcopy(main_net_params)

        mult_ = 11 - 2 * len(vector_field_net_params["channels"])
        vector_field_net = _get_convolution_net(**vector_field_net_params)
        vector_field_net_out_features = \
            vector_field_net_params["channels"][-1] * mult_ * mult_

        features_net = _get_linear_net(**features_net_params)
        features_net_out_features = features_net_params["features"][-1]

        main_net_in_features = \
            vector_field_net_out_features + features_net_out_features
        main_net_params["in_features"] = main_net_in_features
        main_net = _get_linear_net(**main_net_params)

        net = cls(
            features_net=features_net,
            vector_field_net=vector_field_net,
            main_net=main_net
        )

        return net


class StateActionNet(nn.Module):
    def __init__(
        self,
        main_net: nn.Module,
        features_net: nn.Module = None,
        vector_field_net: nn.Module = None,
        action_net: nn.Module = None
    ):
        super().__init__()
        self.main_net = main_net
        self.features_net = features_net
        self.vector_field_net = vector_field_net
        self.action_net = action_net

    def forward(self, state, action):
        features = state["features"]
        vector_field = state["vector_field"]

        batch_size, _, _ = features.shape
        features = features.contiguous().view(batch_size, -1)
        features = self.features_net(features)
        features = features.contiguous().view(batch_size, -1)

        batch_size, _, _, h, w = vector_field.shape
        vector_field = vector_field.contiguous().view(batch_size, -1, h, w)
        vector_field = self.vector_field_net(vector_field)
        vector_field = vector_field.contiguous().view(batch_size, -1)

        action = self.action_net(action)

        x = torch.cat([features, vector_field, action], dim=1)
        x = self.main_net(x)

        return x

    @classmethod
    def get_from_params(
        cls,
        features_net_params=None,
        vector_field_net_params=None,
        action_net_params=None,
        main_net_params=None,
    ) -> "StateActionNet":

        features_net_params = deepcopy(features_net_params)
        vector_field_net_params = deepcopy(vector_field_net_params)
        action_net_params = deepcopy(action_net_params)
        main_net_params = deepcopy(main_net_params)

        mult_ = 11 - 2 * len(vector_field_net_params["channels"])
        vector_field_net = _get_convolution_net(**vector_field_net_params)
        vector_field_net_out_features = \
            vector_field_net_params["channels"][-1] * mult_ * mult_

        features_net = _get_linear_net(**features_net_params)
        features_net_out_features = features_net_params["features"][-1]

        action_net = _get_linear_net(**action_net_params)
        action_net_out_features = action_net_params["features"][-1]

        main_net_in_features = \
            vector_field_net_out_features \
            + features_net_out_features \
            + action_net_out_features
        main_net_params["in_features"] = main_net_in_features
        main_net = _get_linear_net(**main_net_params)

        net = cls(
            features_net=features_net,
            vector_field_net=vector_field_net,
            action_net=action_net,
            main_net=main_net
        )

        return net
