import torch.nn as nn


def get_n_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) *(dilation - 1)
    pad = (kernel_size - 1) // 2
    return pad


class ResUnit(nn.Module):
    def __init__(self, in_channels):
        super(ResUnit, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(get_n_padding(3, 1)),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            nn.ReflectionPad2d(get_n_padding(3, 1)),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        return x + self.block(x)


def up_sample_block(in_channels, out_channels, scale_factor=2):
    block = nn.Sequential(
        nn.ReflectionPad2d(get_n_padding(3, 1)),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels * (scale_factor ** 2), kernel_size=3),
        nn.PixelShuffle(scale_factor),
        nn.PReLU()
    )
    return block


def down_sample_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2)
    )
    return block
