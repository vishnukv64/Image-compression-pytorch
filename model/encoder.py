from model.block import *


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, nf=32, weight_init=True):
        super(Encoder, self).__init__()

        block = []

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(get_n_padding(5, 1)),
            nn.Conv2d(in_channels=in_channels, out_channels=nf, kernel_size=5),
            nn.PReLU()
        )

        for _ in range(3):
            block += [
                ResUnit(in_channels=nf),
                ResUnit(in_channels=nf),
                down_sample_block(nf, nf)
            ]

        block += [
            ResUnit(in_channels=nf),
            ResUnit(in_channels=nf)
        ]

        self.iter_block = nn.Sequential(*block)

        self.out_block = nn.Sequential(
            nn.ReflectionPad2d(get_n_padding(3, 1)),
            nn.Conv2d(in_channels=nf, out_channels=out_channels, kernel_size=3),
            nn.Sigmoid()
        )

        if weight_init:
            self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.iter_block(x)
        x = self.out_block(x)
        # return torch.round(x * 63)
        return x * 63

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
