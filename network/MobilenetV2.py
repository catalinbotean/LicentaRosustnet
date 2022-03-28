from typing import Callable, Any, Optional, List

from torch import Tensor
from torch import nn
from network.mynn import forgiving_state_restore
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from .misc import Conv2dNormActivation
from ._utils import _make_divisible
from .instance_whitening import InstanceWhitening

__all__ = ["MobileNetV2", "mobilenet_v2"]

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        iw: int = 0,
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 insted of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.iw = iw
        self.expand_ratio = expand_ratio
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        if iw == 1:
            self.instance_norm_layer = InstanceWhitening(oup)
        elif iw == 2:
            self.instance_norm_layer = InstanceWhitening(oup)
        elif iw == 3:
            self.instance_norm_layer = nn.InstanceNorm2d(oup, affine=False)
        elif iw == 4:
            self.instance_norm_layer = nn.InstanceNorm2d(oup, affine=False)
        else:
            self.instance_norm_layer = nn.Sequential()
        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x_tuple):
        if len(x_tuple) == 2:
            x = x_tuple[0]
        else:
            print("error in invert residual forward path")
            return
        if self.expand_ratio != 1:
            x_tuple = self.conv[0](x_tuple)
            x_tuple = self.conv[1](x_tuple)
            conv_x = x_tuple[0]
            w_arr = x_tuple[1]
            conv_x = self.conv[2](conv_x)
            conv_x = self.conv[3](conv_x)
        else:
            x_tuple = self.conv[0](x_tuple)
            conv_x = x_tuple[0]
            w_arr = x_tuple[1]
            conv_x = self.conv[1](conv_x)
            conv_x = self.conv[2](conv_x)

        if self.use_res_connect:
            x = x + conv_x
        else:
            x = conv_x

        if self.iw >= 1:
            if self.iw == 1 or self.iw == 2:
                x, w = self.instance_norm_layer(x)
                w_arr.append(w)
            else:
                x = self.instance_norm_layer(x)

        return [x, w_arr]


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        iw: list = [0, 0, 0, 0, 0, 0, 0],
    ) -> None:
        super().__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [
            Conv2dNormActivation(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)
        ]
        feature_count = 0
        iw_layer = [1, 6, 10, 17, 18]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                feature_count += 1
                stride = s if i == 0 else 1
                if feature_count in iw_layer:
                    layer = iw_layer.index(feature_count)
                    features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, iw=iw[layer+2]))
                else:
                    features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, iw=0))
                input_channel = output_channel
        features.append(
            Conv2dNormActivation(
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6
            )
        )
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def mobilenet_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV2:
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        forgiving_state_restore(model, state_dict)
    return model
