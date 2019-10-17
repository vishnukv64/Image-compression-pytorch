import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, num_conv_block=4, weight_init=True):
        super(Discriminator, self).__init__()

        block = []

        in_channels = 3
        out_channels = 64

        for _ in range(num_conv_block):
            block += [
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()

            ]
            in_channels = out_channels

            block += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()

            ]
            out_channels *= 2

        out_channels //= 2
        in_channels = out_channels

        block += [nn.Conv2d(in_channels, out_channels, 3),
                  nn.LeakyReLU(0.2),
                  nn.Conv2d(out_channels, out_channels, 3)]

        self.feature_extraction = nn.Sequential(*block)

        self.classification = nn.Sequential(
            nn.Linear(512 * 9 * 9, 100),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

        if weight_init:
            self._init_weight()

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), -1)
        x = self.classification(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
