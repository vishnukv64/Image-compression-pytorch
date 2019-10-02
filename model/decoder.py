from model.block import *


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, nf=32):
        super(Decoder, self).__init__()

        block = []

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(get_n_padding(3, 1)),
            nn.Conv2d(in_channels=in_channels, out_channels=nf, kernel_size=3),
            nn.PReLU()
        )

        for _ in range(3):
            block += [
                ResUnit(in_channels=nf),
                ResUnit(in_channels=nf),
                up_sample_block(nf, nf)
            ]

        self.iter_block = nn.Sequential(*block)

        self.out_block = nn.Sequential(
            nn.ReflectionPad2d(get_n_padding(5, 1)),
            nn.Conv2d(in_channels=nf, out_channels=out_channels, kernel_size=5),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x / 63
        x = self.conv1(x)
        x = self.iter_block(x)
        x = self.out_block(x)
        return x
