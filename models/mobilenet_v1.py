import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from .slimmable_ops import USBatchNorm2d, USConv2d, USLinear, make_divisible
from utils.config import FLAGS
from utils.dct import *

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, inp, outp, stride):
        super(DepthwiseSeparableConv, self).__init__()
        assert stride in [1, 2]

        layers = [
            USConv2d(
                inp, inp, 3, stride, 1, groups=inp, depthwise=True,
                bias=False),
            USBatchNorm2d(inp),
            nn.ReLU6(inplace=True),

            USConv2d(inp, outp, 1, 1, 0, bias=False),
            USBatchNorm2d(outp),
            nn.ReLU6(inplace=True),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)

class Model(nn.Module):
    def __init__(self, num_classes=1000, input_size=224):
        super(Model, self).__init__()

        # setting of inverted residual blocks
        init_channel = 32
        self.block_setting = [
            # c, n, s
            [64, 1, 1],
            [128, 2, 2],
            [256, 2, 2],
            [512, 6, 2],
            [1024, 2, 2],
        ]

        self.features = []

        # head
        assert input_size % 32 == 0
        channels = make_divisible(init_channel)
        self.outp = make_divisible(1024)
        first_stride = 2
        self.features.append(
            nn.Sequential(
                USConv2d(
                    3, channels, 3, first_stride, 1, bias=False,
                    us=[False, True]),
                USBatchNorm2d(channels),
                nn.ReLU6(inplace=True))
        )

        # body
        for c, n, s in self.block_setting:
            outp = make_divisible(c)
            for i in range(n):
                if i == 0:
                    self.features.append(
                        DepthwiseSeparableConv(channels, outp, s))
                else:
                    self.features.append(
                        DepthwiseSeparableConv(channels, outp, 1))
                channels = outp

        # avg_pool_size = input_size// 32
        # self.features.append(nn.AvgPool2d(avg_pool_size))
        self.features.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # classifier
        self.classifier = nn.Sequential(
            USLinear(self.outp, num_classes, us=[True, False])
        )

        # policy network
        self.pool = nn.AdaptiveMaxPool2d(output_size=(28, 28))
        self.LN = nn.LayerNorm(normalized_shape=[28, 28], elementwise_affine=False)
        self.attn = nn.Conv2d(1, 1, 3, padding=1)
        self.linear = nn.Sequential(
            nn.Linear(28 * 28, 14 * 14),
            nn.ReLU(),
            nn.Linear(14 * 14, len(FLAGS.width_mult_list))
        )

        if FLAGS.reset_parameters:
            self.reset_parameters()

    def forward(self, x, tau=None, policy=False):
        if policy:
            # denormalize and convert to gray scale
            n, c, h, w = x.size()
            x = x.permute(1, 0, 2, 3).reshape(c, -1)
            mean, std, togray = torch.tensor([0.485, 0.456, 0.406]).cuda(non_blocking=True), \
                                torch.tensor([0.229, 0.224, 0.225]).cuda(non_blocking=True), \
                                torch.tensor([0.2989, 0.5870, 0.1140]).cuda(non_blocking=True)
            for t, m, s in zip(x, mean, std):
                t.mul_(s).add_(m)
            x = (x * 255 * togray.unsqueeze(-1)).sum(dim=0, keepdim=True).view(n, 1, h, w)
            # dct, attention, gumbel softmax
            x = self.LN(self.pool(dct_2d(x)))
            attn = torch.sigmoid(self.attn(x))
            x = self.linear((x * attn).reshape(n, -1))
            x = F.gumbel_softmax(F.softmax(x, dim=1), tau=tau, hard=True, dim=1)
        else:
            x = self.features(x)
            last_dim = x.size()[1]
            x = x.view(-1, last_dim)
            x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
