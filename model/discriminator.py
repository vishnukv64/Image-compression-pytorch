import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, num_conv_block=4):
        super(Discriminator, self).__init__()

        block = []

        in_channels = 3
        out_channels = 64

        for _ in range(num_conv_block):
            block += [
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU()

            ]
            in_channels = out_channels

            block += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2),
                nn.InstanceNorm2d(out_channels),
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
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), -1)
        x = self.classification(x)
        return x